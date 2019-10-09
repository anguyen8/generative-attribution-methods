import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch.optim
from matplotlib import cm
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from matplotlib.colors import ListedColormap

use_cuda = torch.cuda.is_available()


# Added for loading Places365 class labels
def load_class_label():
    file_name = './categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    return classes


# Added for loading ImageNet classes
def load_imagenet_label_map():
    """
    Load ImageNet label dictionary.
    return:
    """

    input_f = open("./imagenet_classes.txt")
    label_map = {}
    for line in input_f:
        parts = line.strip().split(": ")
        (num, label) = (int(parts[0]), parts[1].replace('"', ""))
        label_map[num] = label

    input_f.close()
    return label_map


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta).sum()
    col_grad = torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta).sum()
    return row_grad + col_grad


def unnormalize(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] * stds[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] + means[i]
    return preprocessed_img


def normalize(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] * stds[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] + means[i]
    preprocessed_img = np.expand_dims(preprocessed_img, 0)
    return preprocessed_img


def load_model(arch_name='googlenet'):
    if arch_name == 'googlenet':
        model = models.googlenet(pretrained=True)
    elif arch_name == 'inceptionv3':
        model = models.inception_v3(pretrained=True)
    elif arch_name == 'resnet50':
        model = models.resnet50(pretrained=True)

    return model


def load_model_places365(arch_name='resnet50'):

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch_name
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch_name](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model


def preprocess_image(img, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    preprocessed_img_tensor = transform(np.uint8(255 * img))

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = preprocessed_img_tensor.permute(1, 2, 0).numpy()[:, :, ::-1]
    preprocessed_img = (preprocessed_img - means) / stds

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).to('cuda')
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.requires_grad = False
    preprocessed_img_tensor = preprocessed_img_tensor.permute(2, 0, 1)
    preprocessed_img_tensor.unsqueeze_(0)
    preprocessed_img_tensor = preprocessed_img_tensor.float()
    preprocessed_img_tensor.requires_grad = False
    return preprocessed_img_tensor


def load(mask, img, blurred):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    return np.uint8(255 * perturbated), np.uint8(255 * mask)


def mkdir_p(mypath):

    """Creates a directory. equivalent to using mkdir -p on the command line"""

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def clamp(input, min=None, max=None):
    if min is not None and max is not None:
        return torch.clamp(input, min=min, max=max)
    elif min is None and max is None:
        return input
    elif min is None and max is not None:
        return torch.clamp(input, max=max)
    elif min is not None and max is None:
        return torch.clamp(input, min=min)
    else:
        raise ValueError("This is impossible")


def zero_out_plot_multiple_patch(grid,
                  folderName,
                  row_labels_left,
                  row_labels_right,
                  col_labels,
                  file_name=None,
                  dpi=224,
                  ):

    plt.rcParams['axes.linewidth'] = 0.0  # set the value globally
    plt.rcParams.update({'font.size': 5})
    plt.rc("font", family="sans-serif")
    plt.rc("axes.spines", top=True, right=True, left=True, bottom=True)
    image_size = (grid[0][0]).shape[0]
    nRows = len(grid)
    nCols = len(grid[0])
    tRows = nRows + 2  # total rows
    tCols = nCols + 1  # total cols
    wFig = tCols
    hFig = tRows  # Figure height (one more than nRows becasue I want to add xlabels to the top of figure)
    fig, axes = plt.subplots(nrows=tRows, ncols=tCols, figsize=(wFig, hFig))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    axes = np.reshape(axes, (tRows, tCols))
    #########

    # Creating colormap
    uP = cm.get_cmap('Reds', 129)
    dowN = cm.get_cmap('Blues_r', 128)
    newcolors = np.vstack((
        dowN(np.linspace(0, 1, 128)),
        uP(np.linspace(0, 1, 129))
    ))
    cMap = ListedColormap(newcolors, name='RedsBlues')
    cMap.colors[257//2, :] = [1, 1, 1, 1]

    #######
    scale = 0.99
    fontsize = 15
    o_img = grid[0][0]
    for r in range(tRows):
        # if r <= 1:
        for c in range(tCols):
            ax = axes[r][c]
            l, b, w, h = ax.get_position().bounds
            ax.set_position([l, b, w * scale, h * scale])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if r > 0 and c > 0 and r < tRows - 1:
                img_data = grid[r - 1][c - 1]
                abs_min = np.amin(img_data)
                abs_max = np.amax(img_data)
                abs_mx = max(np.abs(abs_min), np.abs(abs_max))
                r_abs_min = round(np.amin(img_data), 2)
                r_abs_max = round(np.amax(img_data), 2)
                r_abs_mx = round(max(np.abs(abs_min), np.abs(abs_max)), 2)

                # Orig Image
                if r == 1 and c == 1:
                    im = ax.imshow(img_data, interpolation='none')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                else:
                    # im = ax.imshow(o_img, interpolation='none', cmap=cMap, vmin=-1, vmax=1)
                    im = ax.imshow(img_data, interpolation='none', cmap=cMap, vmin=-1, vmax=1)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    # save 1

                zero = 0
                if r < tRows:  # not r - 1:
                    if col_labels != []:
                        # ipdb.set_trace()
                        ax.set_xlabel(col_labels[c - 1],
                                      # + '\n' + f'max: {str(r_abs_max)}, min: {str(r_abs_min)}'
                                      horizontalalignment='center',
                                      verticalalignment='bottom',
                                      fontsize=9, labelpad=17)
                if c == tCols - 2:
                    if row_labels_right != []:
                        txt_right = [l + '\n' for l in row_labels_right[r - 1]]
                        ax2 = ax.twinx()
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['bottom'].set_visible(False)
                        ax2.spines['left'].set_visible(False)
                        ax2.set_ylabel(''.join(txt_right), rotation=0,
                                       verticalalignment='center',
                                       horizontalalignment='left',
                                       fontsize=fontsize)
                if c == 1:  # (not c - 1) or (not c - 2) or (not c - 4) or (not c - 6):
                    if row_labels_left != []:
                        txt_left = [l + '\n' for l in row_labels_left[r - 1]]
                        ax.set_ylabel(''.join(row_labels_left[0]),
                                      # rotation=0,
                                      # verticalalignment='center',
                                      # horizontalalignment='center',
                                      fontsize=fontsize)
                # else:
                if c == tCols - 1:  # > 1 # != 1:
                    w_cbar = 0.009
                    h_cbar = h * 0.9  # scale
                    b_cbar = b
                    l_cbar = l + scale * w + 0.001
                    cbaxes = fig.add_axes([l_cbar + 0.015, b_cbar + 0.015, w_cbar, h_cbar])
                    cbar = fig.colorbar(im, cax=cbaxes)
                    cbar.outline.set_visible(False)
                    cbar.ax.tick_params(labelsize=15, width=0.2, length=1.2, direction='inout', pad=0.5)
                    tt = 1
                    cbar.set_ticks([])
                    cbar.set_ticks([-tt, zero, tt])
                    cbar.set_ticklabels([-1, zero, 1])

        #####################################################################################
    dir_path = folderName
    # print(f'Saving figure to {os.path.join(dir_path, file_name)}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(dir_path, file_name), dpi=dpi / scale, transparent=True,
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


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