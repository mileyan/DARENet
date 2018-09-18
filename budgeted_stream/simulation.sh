#!/bin/bash
set -e
dataset_path=$1
feature_path=$2

echo "running simulation based on distance confidence function"
python main.py --log_file distance_confidence --confidence_function distance --dataset_path ${dataset_path} --feature_path ${feature_path} --test_budget

echo "running simulation based on margin confidence function"
python main.py --log_file margin_confidence --confidence_function margin --dataset_path ${dataset_path} --feature_path ${feature_path} --test_budget

echo "running simulation based on random exit"
python main.py --log_file random --confidence_function random --dataset_path ${dataset_path} --feature_path ${feature_path} --test_budget

echo "simulation finished, plotting"
python budgeted_stream_plot.py
echo "all finished!"
