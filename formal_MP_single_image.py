import os
import cv2
import sys
import time
import scipy
import torch
import argparse
import numpy as np
import torch.optim

from formal_utils import *
from skimage.transform import resize
from PIL import ImageFilter, Image

use_cuda = torch.cuda.is_available()

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.to('cuda')  # cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def create_blurred_circular_mask(mask_shape, radius, center=None, sigma=10):
    assert (len(mask_shape) == 2)
    if center is None:
        x_center = int(mask_shape[1] / float(2))
        y_center = int(mask_shape[0] / float(2))
        center = (x_center, y_center)
    y, x = np.ogrid[-y_center:mask_shape[0] - y_center, -x_center:mask_shape[1] - x_center]
    mask = x * x + y * y <= radius * radius
    grid = np.zeros(mask_shape)
    grid[mask] = 1

    if sigma is not None:
        grid = scipy.ndimage.filters.gaussian_filter(grid, sigma)
    return grid


def create_blurred_circular_mask_pyramid(mask_shape, radii, sigma=10):
    assert (len(mask_shape) == 2)
    num_masks = len(radii)
    masks = np.zeros((num_masks, 3, mask_shape[0], mask_shape[1]))
    for i in range(num_masks):
        masks[i, :, :, :] = create_blurred_circular_mask(mask_shape, radii[i], sigma=sigma)
    return masks


def test_circular_masks(args, model, inpaint_model, o_img, upsample, gt_category, radii=np.arange(0, 175, 5),
                        thres=1e-2):

    masks = create_blurred_circular_mask_pyramid((args.size, args.size), radii)
    masks = 1 - masks
    u_mask = upsample(torch.from_numpy(masks)).float().to('cuda')
    num_masks = len(radii)
    img = preprocess_image(np.float32(o_img) / 255, size)

    gradient = np.zeros((1, 1000))
    gradient[0][gt_category] = 1
    scores = np.zeros(num_masks)
    batch_masked_img = []
    for i in range(num_masks):
        if args.algo == 'MP':
            null_img = preprocess_image(get_blurred_img(np.float32(o_img)), args.size)
            masked_img = img.mul(u_mask[i]) + null_img.mul(1 - u_mask[i])
        elif args.algo == 'MPG':
            # Use inpainted image for optimization
            temp_inpaint_img, _ = inpaint_model.generate_background(img, u_mask[i].unsqueeze(0))
            if args.perturb_binary:
                thresh = max(0.5, args.thresh * (torch.max(u_mask[i]).cpu().item() + torch.min(
                    u_mask[i]).cpu().item()))
                u_mask[i].data = torch.where(u_mask[i].data > thresh,
                                             torch.ones_like(u_mask[i].data),
                                             torch.zeros_like(u_mask[i].data))
            masked_img = img.mul(u_mask[i]) + temp_inpaint_img.mul(1 - u_mask[i])
        else:
            print('Invalid heatmap style!!')
            exit(0)

        outputs = torch.nn.Softmax(dim=1)(model(masked_img))
        scores[i] = outputs[0, gt_category].cpu().detach()
        batch_masked_img.append(masked_img)
    img_output = torch.nn.Softmax(dim=1)(model(img)).cpu().detach()
    orig_score = img_output[0, gt_category]

    percs = (scores - scores[-1]) / float(orig_score - scores[-1])
    try:
        first_i = np.where(percs < thres)[0][0]
    except:
        first_i = -1
    return radii[first_i]


def get_blurred_img(img, radius=10):
    img = Image.fromarray(np.uint8(img))
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.array(blurred_img) / float(255)


if __name__ == '__main__':

    # Hyper parameters.
    parser = argparse.ArgumentParser(description='Processing Meaningful Perturbation data')
    parser.add_argument('--img_path', type=str,
                        default='/home/chirag/ILSVRC2012_img_val_bb/ILSVRC2012_img_val/',
                        help='filepath for the example image')

    parser.add_argument('--algo', type=str,
                        default='MP', help='MP|MPG')

    parser.add_argument('--mask_init', type=str,
                        default='random', help='random|circular')

    parser.add_argument('--perturb_binary', type=int,
                        default=0,
                        help='flag for using binary mask just for perturbation')

    parser.add_argument('--learning_rate', type=float,
                        default=0.1,
                        help='flag for using binary mask just for perturbation')

    parser.add_argument('--size', type=int,
                        default=224, help='mask size to be optimized')

    parser.add_argument('--true_class', type=int,
                        default=565,
                        help='target class of the image you want to explain')

    parser.add_argument('--num_iter', type=int,
                        default=300, help='enter number of optimization iterations')

    parser.add_argument('--jitter', type=int,
                        default=4, help='jitter')

    parser.add_argument('--l1_coeff', type=float,
                        default=1e-4, help='L1 coefficient regularizer')

    parser.add_argument('--tv_coeff', type=float,
                        default=1e-2, help='TV coefficient regularizer')

    parser.add_argument('--thresh', type=float,
                        default=0.5, help='threshold for binarizing mask')

    parser.add_argument('--dataset', type=str,
                        default='imagenet',
                        help='dataset to run on imagenet | places365')

    parser.add_argument('--save_path', type=str,
                        default='./',
                        help='filepath for the example image')

    parser.add_argument('--weight_file', type=str,
                        default='/home/chirag/gpu3_codes/generative_inpainting_FIDO/model_logs/release_imagenet_256/',
                        help='path for the weight files of the inpainter model for imagenet | places365')
    args = parser.parse_args()

    # PyTorch random seed
    torch.manual_seed(0)

    tv_beta = 3
    learning_rate = args.learning_rate
    max_iterations = args.num_iter
    l1_coeff = args.l1_coeff
    tv_coeff = args.tv_coeff
    size = args.size

    if args.dataset == 'imagenet':
        model = load_model(arch_name='resnet50')

    elif args.dataset == 'places365':
        model = load_model_places365(arch_name='resnet50')

    else:
        print('Invalid datasest!!')
        exit(0)

    model = torch.nn.DataParallel(model).to('cuda')
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    if args.algo == 'MPG':
        # Tensorflow CA-inpainter from FIDO
        sys.path.insert(0, './generative_inpainting')
        from CAInpainter import CAInpainter

        inpaint_model = CAInpainter(1, checkpoint_dir=args.weight_file)

    if use_cuda:
        upsample = torch.nn.UpsamplingNearest2d(size=(size, size)).to('cuda')

    else:
        upsample = torch.nn.UpsamplingNearest2d(size=(size, size))

    init_time = time.time()

    # Read image
    original_img = cv2.imread(args.img_path, 1)

    shape = original_img.shape
    img = np.float32(original_img) / 255

    gt_category = args.true_class

    # define jitter function
    jitter = args.jitter

    # Convert to torch variables
    img = preprocess_image(img, size + jitter)  # img

    if use_cuda:
        img = img.to('cuda')

    # Path to the output folder
    save_path = os.path.join(args.save_path, '{}_{}'.format(args.algo, str(args.perturb_binary)),
                             '{}'.format(args.dataset))

    if not os.path.isdir(os.path.join(save_path)):
        mkdir_p(os.path.join(save_path))

    # Modified
    if args.mask_init == 'random':
        np.random.seed(seed=0)
        mask = np.random.rand(28, 28)
        mask = numpy_to_torch(mask)
    elif args.mask_init == 'circular':

        # CAFFE mask_init
        if args.algo == 'MP':
            mask_radius = test_circular_masks(args, model, model, original_img, upsample, gt_category)
        elif args.algo == 'MPG':
            mask_radius = test_circular_masks(args, model, inpaint_model, original_img, upsample, gt_category)
        mask = 1 - create_blurred_circular_mask((size, size), mask_radius, center=None, sigma=10)
        mask = resize(mask.astype(float), (size, size))
        mask = numpy_to_torch(mask)
    else:
        print('Invalid mask init!!')
        exit(0)

    if args.algo == 'MP':
        null_img = preprocess_image(get_blurred_img(np.float32(original_img), radius=10), size + jitter)

    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    l1 = []
    l2 = []
    l3 = []
    for i in range(max_iterations):
        if jitter != 0:
            j1 = np.random.randint(jitter)
            j2 = np.random.randint(jitter)
        else:
            j1 = 0
            j2 = 0

        upsampled_mask = upsample(mask)

        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))

        if args.algo == 'MPG':
            # Tensorflow CA-inpainter
            inpaint_img, _ = inpaint_model.generate_background(img[:, :, j1:(size + j1), j2:(size + j2)],
                                                              upsampled_mask)

            if args.perturb_binary:
                thresh = max(0.5, args.thresh * (torch.max(upsampled_mask).cpu().item() + torch.min(
                    upsampled_mask).cpu().item()))
                upsampled_mask.data = torch.where(upsampled_mask.data > thresh,
                                                  torch.ones_like(upsampled_mask.data),
                                                  torch.zeros_like(upsampled_mask.data))
                perturbated_input = img[:, :, j1:(size + j1), j2:(size + j2)].mul(upsampled_mask) + \
                                    inpaint_img.mul(1 - upsampled_mask)
            else:
                perturbated_input = img[:, :, j1:(size + j1), j2:(size + j2)].mul(upsampled_mask) + \
                                    inpaint_img.mul(1 - upsampled_mask)

        elif args.algo == 'MP':
            if args.perturb_binary:
                thresh = max(0.5, args.thresh * (torch.max(upsampled_mask).cpu().item() + torch.min(
                    upsampled_mask).cpu().item()))
                upsampled_mask.data = torch.where(upsampled_mask.data > thresh,
                                                  torch.ones_like(upsampled_mask.data),
                                                  torch.zeros_like(upsampled_mask.data))
                perturbated_input = img[:, :, j1:(size + j1), j2:(size + j2)].mul(upsampled_mask) + \
                                    null_img[:, :, j1:(size + j1), j2:(size + j2)].mul(
                                        1 - upsampled_mask)
            else:
                perturbated_input = img[:, :, j1:(size + j1), j2:(size + j2)].mul(upsampled_mask) + \
                                    null_img[:, :, j1:(size + j1), j2:(size + j2)].mul(
                                        1 - upsampled_mask)

        else:
            print('Invalid heatmap style!!')
            exit(0)

        optimizer.zero_grad()
        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))

        loss = l1_coeff * torch.sum(torch.abs(1 - mask)) + tv_coeff * tv_norm(mask, tv_beta) + \
               outputs[0, gt_category]
        loss.backward()

        optimizer.step()
        mask.data.clamp_(0, 1)

    np.save(os.path.join(save_path, "mask_{}.npy".format(args.algo)),
            1 - mask.cpu().detach().numpy()[0, 0, :])

    # Normalize the attribution map for visualization purpose
    mask = 1 - (mask - mask.min()) / (mask.max() - mask.min())
    cv2.imwrite(os.path.join(save_path, "mask_{}.png".format(args.algo)),
                cv2.applyColorMap((resize(mask.cpu().detach().numpy()[0, 0, :],
                                          (args.size, args.size)) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS))

    # save original image
    pill_transf = get_pil_transform()
    cv2.imwrite(os.path.join(save_path, "original.png"), cv2.cvtColor(np.array(pill_transf(get_image(args.img_path))),
                                                                      cv2.COLOR_BGR2RGB))

    # print('Time taken: {:.3f}'.format(time.time() - init_time))
