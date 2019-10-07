#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# 2019

img_path='teaser_image.JPEG'
true_class=565
dataset='imagenet'
weight_file='./generative_inpainting/model_logs/release_imagenet_256/'
save_path='./output/'
algo='LIME'

# LIME
CUDA_VISIBLE_DEVICES=0 python formal_LIME_single_image.py --img_path ${img_path} --true_class ${true_class} --dataset ${dataset} --weight_file ${weight_file} --save_path ${save_path} --algo ${algo}

echo "### Output for LIME ###"
montage -quiet ${save_path}/${algo}/${dataset}/original.png ${save_path}/${algo}/${dataset}/mask_${algo}.png -tile x1 -geometry +2+2 ${save_path}/${algo}/${dataset}/out_${algo}.jpg
imgcat ${save_path}/${algo}/${dataset}/out_${algo}.jpg

# LIME-G
algo='LIMEG'
CUDA_VISIBLE_DEVICES=0 python formal_LIME_single_image.py --img_path ${img_path} --true_class ${true_class} --dataset ${dataset} --weight_file ${weight_file} --save_path ${save_path} --algo ${algo}

echo "### Output for LIME-G ###"
montage -quiet ${save_path}/${algo}/${dataset}/original.png ${save_path}/${algo}/${dataset}/mask_${algo}.png -tile x1 -geometry +2+2 ${save_path}/${algo}/${dataset}/out_${algo}.jpg
imgcat ${save_path}/${algo}/${dataset}/out_${algo}.jpg
