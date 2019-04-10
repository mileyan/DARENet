#!/bin/bash
set -e
nettype=$1
GPU=$2
dataset_path=$3
dataset=$4
checkpoint_name=$5
feature_path=$6
gen_stage_features=$7

if [[ ${gen_stage_features} == 'True' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python main.py --arch ${nettype}  --dataset ${dataset} -j 4  -b 64 --crop_size 256 128 \
    --data ${dataset_path} --extract_features --resume ./checkpoints/$checkpoint_name/checkpoint.pth.tar \
    --extract_features_folder ${feature_path} --ten_crop
else
    CUDA_VISIBLE_DEVICES=${GPU} python main.py --arch ${nettype}  --dataset ${dataset} -j 4  -b 64 --crop_size 256 128 \
    --data ${dataset_path} --extract_features --resume ./checkpoints/$checkpoint_name/checkpoint.pth.tar \
    --extract_features_folder ${feature_path} --ten_crop
fi
