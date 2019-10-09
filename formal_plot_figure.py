import os
import cv2
import ipdb
import random
import argparse
import numpy as np
from formal_utils import *
import matplotlib.pyplot as plt
from skimage.transform import resize
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10, 'font.weight':'bold'})
plt.rc("font", family="sans-serif")


if __name__ == '__main__':

    # Hyper parameters.
    parser = argparse.ArgumentParser(description='Processing Meaningful Perturbation data')
    parser.add_argument('--result_path',
                        default='./output/',
                        type=str, help='filepath for the results')
    parser.add_argument('--algo',
                        default='MP',
                        type=str, help='SP|SPG|LIME|LIMEG|MP|MPG')
    parser.add_argument('--dataset',
                        default='imagenet',
                        type=str, help='dataset to run on imagenet | places365')
    parser.add_argument('--save_path',
                        default='',
                        type=str, help='path for saving images')

    args = parser.parse_args()

    # result_path ./output/SP/

    # Read real image
    o_img_path = [f for f in os.listdir(os.path.join(args.result_path, args.dataset)) if f.startswith('real')][0]
    o_img = cv2.cvtColor(cv2.imread(os.path.join(args.result_path, args.dataset, o_img_path), 1), cv2.COLOR_BGR2RGB)
    labels = [' '.join(o_img_path.split('_')[1:3])]

    # Read generated heatmap
    heatmap_path = [f for f in os.listdir(os.path.join(args.result_path, args.dataset)) if f.endswith('.npy')][0]
    heatmap = resize(np.load(os.path.join(args.result_path, args.dataset, heatmap_path)), (224, 224))

    # Read intermediate perturbed images
    intermediate_path = [j for j in os.listdir(os.path.join(args.result_path, args.dataset, 'intermediate_steps'))
                         if j.startswith('intermediate_')]
    random.seed(0)
    intermediate_path = random.choices(intermediate_path, k=5)
    intermediate = [cv2.cvtColor(cv2.imread(os.path.join(args.result_path, args.dataset, 'intermediate_steps', j), 1),
                                 cv2.COLOR_BGR2RGB) for j in intermediate_path]
    labels.extend([' '.join(j.split('_')[2:4]) for j in intermediate_path])
    labels.extend([''])

    # Make a list of all images to be plotted
    image_batch = [o_img]
    image_batch.extend(intermediate)
    image_batch.extend([heatmap])
    zero_out_plot_multiple_patch([image_batch],
                                 folderName='./',
                                 row_labels_left=[args.algo],
                                 row_labels_right=[],
                                 col_labels=labels,
                                 file_name=os.path.join(args.save_path, 'figure_{}.jpg'.format(args.algo)))
