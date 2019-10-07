#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# 2019

img_path='teaser_image.JPEG'
true_class=565
dataset='imagenet'
weight_file='./generative_inpainting/model_logs/release_imagenet_256/'
save_path='./output/'
algo='SP'
patch_size=29
stride=3

# SP
CUDA_VISIBLE_DEVICES=0 python formal_SP_single_image.py --img_path ${img_path} --true_class ${true_class} --dataset ${dataset} --weight_file ${weight_file} --save_path ${save_path} --algo ${algo} --patch_size ${patch_size} --stride ${stride}

echo "### Output for SP ###"
montage -quiet ${save_path}/${algo}/${dataset}/original.png $(ls ${save_path}/${algo}/${dataset}/intermediate_steps/* | sort -R | tail -5) ${save_path}/${algo}/${dataset}/mask_${algo}.png -tile x1 -geometry +2+2 ${save_path}/${algo}/${dataset}/out_${algo}.jpg
imgcat ${save_path}/${algo}/${dataset}/out_${algo}.jpg

# LIME-G
algo='SPG'
CUDA_VISIBLE_DEVICES=0 python formal_SP_single_image.py --img_path ${img_path} --true_class ${true_class} --dataset ${dataset} --weight_file ${weight_file} --save_path ${save_path} --algo ${algo} --patch_size ${patch_size} --stride ${stride}

echo "### Output for SP-G ###"
montage -quiet ${save_path}/${algo}/${dataset}/original.png $(ls ${save_path}/${algo}/${dataset}/intermediate_steps/* | sort -R | tail -5) ${save_path}/${algo}/${dataset}/mask_${algo}.png -tile x1 -geometry +2+2 ${save_path}/${algo}/${dataset}/out_${algo}.jpg
imgcat ${save_path}/${algo}/${dataset}/out_${algo}.jpg
