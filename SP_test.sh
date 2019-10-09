#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# 2019

img_path='teaser_image.JPEG'
true_class=565
dataset='imagenet'
weight_file='./generative_inpainting/model_logs/release_imagenet_256/'
save_path='./output/'
algo_1='SP'
patch_size=29
stride=3

# SP
CUDA_VISIBLE_DEVICES=0 python formal_SP_single_image.py --img_path ${img_path} --true_class ${true_class} --dataset ${dataset} --weight_file ${weight_file} --save_path ${save_path} --algo ${algo_1} --patch_size ${patch_size} --stride ${stride}

montage -quiet ${save_path}/${algo_1}/${dataset}/original.png $(ls ${save_path}/${algo_1}/${dataset}/intermediate_steps/* | sort -R | tail -5) ${save_path}/${algo_1}/${dataset}/mask_${algo_1}.png -tile x1 -geometry +2+2 ${save_path}/${algo_1}/${dataset}/out_${algo_1}.jpg

# SP-G
algo_2='SPG'
CUDA_VISIBLE_DEVICES=0 python formal_SP_single_image.py --img_path ${img_path} --true_class ${true_class} --dataset ${dataset} --weight_file ${weight_file} --save_path ${save_path} --algo ${algo_2} --patch_size ${patch_size} --stride ${stride}

montage -quiet ${save_path}/${algo_2}/${dataset}/original.png $(ls ${save_path}/${algo_2}/${dataset}/intermediate_steps/* | sort -R | tail -5) ${save_path}/${algo_2}/${dataset}/mask_${algo_2}.png -tile x1 -geometry +2+2 ${save_path}/${algo_2}/${dataset}/out_${algo_2}.jpg
imgcat ${save_path}/${algo_1}/${dataset}/out_${algo_1}.jpg
imgcat ${save_path}/${algo_2}/${dataset}/out_${algo_2}.jpg
