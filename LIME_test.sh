#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# 2019

img_path='example.JPEG'
true_class=222
dataset='imagenet'
weight_file='./generative_inpainting/model_logs/release_imagenet_256/'
save_path='./output/'
algo_1='LIME'

# LIME
CUDA_VISIBLE_DEVICES=0 python formal_LIME_single_image.py --img_path ${img_path} --true_class ${true_class} --dataset ${dataset} --weight_file ${weight_file} --save_path ${save_path} --algo ${algo_1}

# Save figure
python formal_plot_figure.py --result_path ${save_path}/${algo_1} --dataset ${dataset} --save_path ${save_path}/${algo_1} --algo ${algo_1}

convert ${save_path}/${algo_1}/figure_${algo_1}.jpg -trim ${save_path}/${algo_1}/figure_${algo_1}.jpg

# LIME-G
algo_2='LIMEG'
CUDA_VISIBLE_DEVICES=0 python formal_LIME_single_image.py --img_path ${img_path} --true_class ${true_class} --dataset ${dataset} --weight_file ${weight_file} --save_path ${save_path} --algo ${algo_2}

# Save figure
python formal_plot_figure.py --result_path ${save_path}/${algo_2} --dataset ${dataset} --save_path ${save_path}/${algo_2} --algo ${algo_2}

convert ${save_path}/${algo_2}/figure_${algo_2}.jpg -trim ${save_path}/${algo_2}/figure_${algo_2}.jpg

# Displaying figure
montage -quiet ${save_path}/${algo_1}/figure_${algo_1}.jpg ${save_path}/${algo_2}/figure_${algo_2}.jpg -tile 1x -geometry +2+2 ${save_path}/test_LIME.jpg
imgcat ${save_path}/test_LIME.jpg
