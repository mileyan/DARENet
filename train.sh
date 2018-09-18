#!/bin/bash
set -e
nettype=$1
GPU=$2
train_dataset_path=$3
checkpoint_name=$4
dataset=$5

CUDA_VISIBLE_DEVICES=${GPU} python main.py --arch ${nettype} --dataset ${dataset} -j 8 -b 1 --lr 3e-4 --weight-decay 1e-4 \
    --crop_size 256 128 --data ${train_dataset_path} --checkpoint_folder ./checkpoint/${checkpoint_name} \
    --mean_loss --margin 0. --pretrained --log_path ./logs/${checkpoint_name}  --max_iter 60000 --lr_decay_point 30000 --random_mask
