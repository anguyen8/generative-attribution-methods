import os
import cv2
import sys
import time
import torch
import argparse
import torch.optim
import numpy as np
from formal_utils import *
from skimage.util import view_as_windows
from skimage.transform import resize
from torch.utils.data import Dataset


use_cuda = torch.cuda.is_available()

# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class occlusion_analysis:
    def __init__(self, image, net, num_classes=256, img_size=227, batch_size=64,
                 org_shape=(224, 224)):
        self.image = image
        self.model = net
        self.num_classes = num_classes
        self.img_size = img_size
        self.org_shape = org_shape
        self.batch_size = batch_size

    def explain(self, neuron, loader, l_map, heatmap_type='SP', path='./'):

        # Compute original output
        org_softmax = torch.nn.Softmax(dim=1)(self.model(self.image))

        eval0 = org_softmax.data[0, neuron]

        batch_heatmap = torch.Tensor().to('cuda')

        # Create save_path for storing intermediate steps
        path = os.path.join(path, 'intermediate_steps')
        mkdir_p(path)

        for i, data in enumerate(loader):
            data = data.to('cuda')
            if heatmap_type == 'SP':
                softmax_out = torch.nn.Softmax(dim=1)(self.model(data * self.image))
                delta = eval0 - softmax_out.data[:, neuron]
                amax, aind = softmax_out.max(dim=1)
                gt_val = softmax_out.data[:, neuron]

                # For saving intermediate steps
                for j in range(data.shape[0])[:1]:
                    temp_img = np.uint8(255 * unnormalize(
                        np.moveaxis((data[j, :] * self.image[0, :]).cpu().detach().numpy().transpose(), 0, 1)))
                    temp_img = add_text(temp_img, 'PT: {}({:.3f})'.format(l_map[aind[j].item()].split(",")[0],
                                                                              amax[j].item()), x_pt=25,
                                        scale=1, size=0.50)
                    temp_img = add_text(temp_img,
                                        'GT:{}({:.3f})'.format(l_map[neuron].split(",")[0], gt_val[j].item()), x_pt=25,
                                        scale=1, size=0.50)
                    cv2.imwrite(
                        os.path.join(path, 'intermediate_{:05d}.png'.format(i * self.batch_size + j)),
                        cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))

            elif heatmap_type == 'SPG':
                inpaint_img, _ = impant_model.generate_background(self.image, data, batch_process=True)
                # inpaint_img = self.image.to(data.device) * data + inpaint_img.to(data.device) * (1 - data)
                inpaint_img = self.image * data + inpaint_img * (1 - data)
                softmax_out = torch.nn.Softmax(dim=1)(self.model(inpaint_img))
                delta = eval0 - softmax_out.data[:, neuron]
                amax, aind = softmax_out.max(dim=1)
                gt_val = softmax_out.data[:, neuron]

                # For saving intermediate steps
                for j in range(inpaint_img.shape[0])[:1]:
                    temp_img = np.uint8(255 * unnormalize(
                        np.moveaxis(inpaint_img[j, :].cpu().detach().numpy().transpose(), 0, 1)))
                    temp_img = add_text(temp_img, 'PT: {}({:.3f})'.format(label_map[aind[j].item()].split(",")[0],
                                                                          amax[j].item()), x_pt=25, scale=1, size=0.50)
                    temp_img = add_text(temp_img,
                                        'GT:{}({:.3f})'.format(label_map[neuron].split(",")[0],
                                                               gt_val[j].item()), x_pt=25, scale=1, size=0.50)
                    cv2.imwrite(
                        os.path.join(path, 'intermediate_{:05d}.png'.format(i * self.batch_size + j)),
                        cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))

            batch_heatmap = torch.cat((batch_heatmap, delta))

        sqrt_shape = len(loader)
        attribution = np.reshape(batch_heatmap.cpu().numpy(), (sqrt_shape, sqrt_shape))

        return attribution


if __name__ == '__main__':

    # Hyper parameters
    parser = argparse.ArgumentParser(description='Processing Meaningful Perturbation data')
    parser.add_argument('--img_path', type=str,
                        default='/home/chirag/ILSVRC2012_img_val_bb/ILSVRC2012_img_val/',
                        help='filepath for the example image')

    parser.add_argument('--algo', type=str,
                        default='SPG', help='SP|SPG')

    parser.add_argument('--size', type=int,
                        default=224, help='mask size to be optimized')

    parser.add_argument('--patch_size', type=int,
                        default=77, help='patch size for occlusion')

    parser.add_argument('--stride', type=int,
                        default=3, help='stride size for occlusion')

    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size')

    parser.add_argument('--true_class', type=int,
                        default=565,
                        help='target class of the image you want to explain')

    parser.add_argument('--dataset', type=str,
                        default='imagenet', help='dataset to run on imagenet | places365')

    parser.add_argument('--save_path', type=str,
                        default='./',
                        help='path for saving results')

    parser.add_argument('--weight_file', type=str,
                        default='/home/chirag/gpu3_codes/generative_inpainting_FIDO/model_logs/release_imagenet_256/',
                        help='path for the weight files of the inpainter model for imagenet | places365')
    args = parser.parse_args()

    if args.dataset == 'imagenet':

        model = load_model(arch_name='resnet50')

        # load the class label
        label_map = load_imagenet_label_map()

    elif args.dataset == 'places365':

        model = load_model_places365(arch_name='resnet50')

        # load the class label
        label_map = load_class_label()
    else:
        print('Invalid datasest!!')
        exit(0)

    model = torch.nn.DataParallel(model).to('cuda')
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    batch_size = int((224 - args.patch_size) / args.stride) + 1

    # Create all occlusion masks initially to save time
    # Create mask
    input_shape = (3, args.size, args.size)
    total_dim = np.prod(input_shape)
    index_matrix = np.arange(total_dim).reshape(input_shape)
    idx_patches = view_as_windows(index_matrix, (3, args.patch_size, args.patch_size), args.stride).reshape((-1,) +
                                                                                                            (3,
                                                                                                             args.patch_size,
                                                                                                             args.patch_size))

    # Start perturbation loop
    batch_size = int((args.size - args.patch_size) / args.stride) + 1
    batch_mask = torch.zeros(((idx_patches.shape[0],) + input_shape), device='cuda')
    total_dim = np.prod(input_shape)
    for i, p in enumerate(idx_patches):
        mask = torch.ones(total_dim, device='cuda')
        mask[p.reshape(-1)] = 0  # occ_val
        batch_mask[i] = mask.reshape(input_shape)

    trainloader = torch.utils.data.DataLoader(batch_mask.cpu(), batch_size=batch_size, shuffle=False,
                                              num_workers=0)
    del mask
    del batch_mask

    if args.algo == 'SPG':

        # Tensorflow CA-inpainter from FIDO
        sys.path.insert(0, './generative_inpainting/')
        from CAInpainter import CAInpainter

        impant_model = CAInpainter(batch_size, checkpoint_dir=args.weight_file)

    init_time = time.time()

    original_img = cv2.imread(args.img_path, 1)

    shape = original_img.shape
    img = np.float32(original_img) / 255

    gt_category = args.true_class

    # Convert to torch variables
    img = preprocess_image(img, args.size)

    if use_cuda:
        img = img.to('cuda')

    # Path to the output folder
    save_path = os.path.join(args.save_path, '{}'.format(args.algo), '{}'.format(args.dataset))
    mkdir_p(save_path)

    # save original image
    pill_transf = get_pil_transform()
    temp_img = add_text(np.array(pill_transf(get_image(args.img_path))), 'Real', x_pt=100, scale=1, size=0.50,
                        text_patch=50)
    cv2.imwrite(os.path.join(save_path, "original.png"), cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))

    t1 = time.time()
    with torch.no_grad():
        # Occlusion class
        heatmap_occ = occlusion_analysis(img, net=model, num_classes=1000, img_size=args.size,
                                         batch_size=batch_size, org_shape=shape)
        for stride in [args.stride]:
            for p_size in [args.patch_size]:
                heatmap = heatmap_occ.explain(neuron=gt_category, loader=trainloader,
                                              heatmap_type=args.algo, path=save_path, l_map=label_map)
                np.save(
                    os.path.join(save_path, 'mask_{}.npy'.format(args.algo)),
                    heatmap)

                # Normalize the attribution map for visualization purpose
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                heatmap = (resize(heatmap, (args.size, args.size)) * 255).astype(np.uint8)

                # Heatmap
                temp_img = add_text(np.stack((heatmap, )*3, axis=2),
                                    'Heatmap', x_pt=100, scale=1, size=0.50, text_patch=50)
                cv2.imwrite(os.path.join(save_path, "mask_{}.png".format(args.algo)), temp_img)

    # print('Time taken: {:.3f}'.format(time.time() - init_time))
