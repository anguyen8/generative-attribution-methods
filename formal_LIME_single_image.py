from __future__ import absolute_import
import warnings
warnings.simplefilter('ignore')

import time, os, sys, cv2, time, argparse
import torch
import random
import numpy as np
from formal_utils import *
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

use_cuda = torch.cuda.is_available()
# Fixing for deterministic results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('--img_path', type=str,
                        help='path of the image you want to explain')

    parser.add_argument('--if_pre', type=int, choices=range(2),
                        help='It is clear from name. Default: Post (0)', default=0,
                        )

    parser.add_argument('--lime_background_pixel', type=int,
                        help='Background pixel for lime to be used for absence of super-pixel. Default=0', default=0,
                        )

    parser.add_argument('--lime_superpixel_num', type=int,
                        help='Number of super pixels used by Lime. Default=50', default=50,
                        )

    parser.add_argument('--lime_num_samples', type=int,
                        help='Number of samples used by Lime. Default=1000', default=500,
                        )

    parser.add_argument('--lime_superpixel_seed', type=int,
                        help='Seed to create random samples for Lime. Default=0', default=0,
                        )

    parser.add_argument('--lime_explainer_seed', type=int,
                        help='Seed to creating Lime explainer. Default=0', default=0,
                        )

    parser.add_argument('--batch_size', type=int,
                        default=10, help='batch size')

    parser.add_argument('--true_class', type=int,
                        default=852,
                        help='target class of the image you want to explain')

    parser.add_argument('--save_path', type=str,
                        default='./',
                        help='filepath for the example image')

    parser.add_argument('--weight_file', type=str,
                        default='/home/chirag/gpu3_codes/generative_inpainting_FIDO/model_logs/release_imagenet_256/',
                        help='path for the weight files of the inpainter model for imagenet | places365')

    parser.add_argument('--dataset', type=str,
                        default='imagenet', help='dataset to run on imagenet | places365')

    parser.add_argument('--algo', type=str,
                        default='LIME', help='fill using lime_background_pixel or inpaint')

    # Parse the arguments
    args = parser.parse_args()

    return args


def load_orig_imagenet_model(arch_name='resnet50', if_pre=0):

    model = models.resnet50(pretrained=True)
    if if_pre == 1:
        pass
    else:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    for p in model.parameters():
        p.requires_grad = False

    model = model.to('cuda')
    model.eval()
    return model


def load_orig_places365_model(arch_name='resnet50', if_pre=0):  #

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch_name
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch_name](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    if if_pre == 1:
        pass
    else:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    for p in model.parameters():
        p.requires_grad = False

    model = model.to('cuda')
    model.eval()
    return model


def get_pytorch_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transf


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()

    if args.dataset == 'imagenet':
        pytorch_model = load_orig_imagenet_model(arch_name='resnet50')

        # load the class label
        label_map = load_imagenet_label_map()

    elif args.dataset == 'places365':
        pytorch_model = load_orig_places365_model(arch_name='resnet50')

        # load the class label
        label_map = load_class_label()

    else:
        print('Invalid datasest!!')
        exit(0)

    pytorch_explainer = lime_image.LimeImageExplainer(random_state=args.lime_explainer_seed)
    slic_parameters = {'n_segments': args.lime_superpixel_num, 'compactness': 30, 'sigma': 3}
    segmenter = SegmentationAlgorithm('slic', **slic_parameters)
    pill_transf = get_pil_transform()

    #########################################################
    # Function to compute probabilities
    # Pytorch
    pytorch_preprocess_transform = get_pytorch_preprocess_transform()

    def pytorch_batch_predict(images):
        batch = torch.stack(tuple(pytorch_preprocess_transform(i) for i in images), dim=0)
        batch = batch.to('cuda')

        if args.if_pre == 1:
            logits = pytorch_model(batch)
            probs = F.softmax(logits, dim=1)
        else:
            probs = pytorch_model(batch)
        return probs

    # Initialize CA-inpainter only for LIMEG
    if args.algo == 'LIMEG':

        # Generative ImageNet Contextual Attention (TENSORFLOW)
        sys.path.insert(0, './generative_inpainting/')
        from CAInpainter import CAInpainter

        inpaint_model = CAInpainter(args.batch_size,
                                    checkpoint_dir=args.weight_file)
        inpaint_model.eval()
    else:
        inpaint_model = pytorch_model

    # Preprocess transform
    pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    random.seed(0)
    init_time = time.time()

    # This image will be passed to Lime Explainer
    img = get_image(args.img_path)

    pytorch_img = pytorch_preprocessFn(Image.open(args.img_path).convert('RGB')).to('cuda').unsqueeze(0)
    outputs = pytorch_model(pytorch_img)

    if args.dataset == 'imagenet':
        true_class = args.true_class
        top_labels = 1
        labels = (true_class,)
    elif args.dataset == 'places365':
        true_class = args.true_class
        top_labels = 5
        labels = (true_class, )

    # LIME analysis

    # save_dir
    save_path = os.path.join(args.save_path, '{}'.format(args.algo), '{}'.format(args.dataset))
    mkdir_p(save_path)
    # save path for intermediate steps
    save_intermediate = os.path.join(save_path, 'intermediate_steps')
    mkdir_p(save_intermediate)

    lime_img = np.array(pill_transf(img))
    t1 = time.time()
    pytorch_lime_explanation = pytorch_explainer.explain_instance(lime_img, pytorch_img, inpaint_model,
                                                                  pytorch_batch_predict,
                                                                  batch_size=args.batch_size,
                                                                  segmentation_fn=segmenter,
                                                                  top_labels=None, labels=labels,
                                                                  hide_color=None,
                                                                  num_samples=args.lime_num_samples,
                                                                  random_seed=args.lime_superpixel_seed,
                                                                  fill_type=args.algo,
                                                                  num_super_pixel=args.lime_superpixel_num,
                                                                  sav_path=save_intermediate,
                                                                  target_category=true_class, l_map=label_map)
    pytorch_segments = pytorch_lime_explanation.segments
    pytorch_heatmap = np.zeros(pytorch_segments.shape)
    local_exp = pytorch_lime_explanation.local_exp
    exp = local_exp[true_class]

    for i, (seg_idx, seg_val) in enumerate(exp):
        pytorch_heatmap[pytorch_segments == seg_idx] = seg_val

    # print('Time taken: {:.3f} secs'.format(time.time()-init_time))

    # SAVE raw numpy values
    np.save(os.path.join(save_path, "mask_{}.npy".format(args.algo)), pytorch_heatmap)

    # Compute original output
    org_softmax = pytorch_model(pytorch_img)
    eval0 = org_softmax.data[0, true_class]
    pill_transf = get_pil_transform()
    cv2.imwrite(os.path.join(save_path, 'real_{}_{:.3f}_image.jpg'
                             .format(label_map[true_class].split(',')[0].split(' ')[0].split('-')[0], eval0)),
                cv2.cvtColor(np.array(pill_transf(get_image(args.img_path))), cv2.COLOR_BGR2RGB))