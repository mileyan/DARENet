# Budgeted Stream Simulation

Code base for simulation on the budgeted stream person re-ID for paper *Resource Aware Person Re-identification across Multiple Resolutions*.

This code generates Figure 6 in the paper.

## Requirement
* python
* numpy
* colorlog (for logging)
* scipy (for reading `.mat` file)
* matplotlib (for plotting)
* seaborn (for plotting)

## How to run
### Data
**Feature vectors**: You should use the trained network to generate feature vectores for query and gallery images at stage 1-3 and fusion stage. We use `.mat` file type in this code. Name them as `query_features_[1-3|fusion].mat` and `test_features_[1-3|fusion].mat` and store them in a folder (e.g. `./data/feature/DaRe`)

**Original data**: Original data are needed for collecting data labels. You can download `Market-1501-v15.09.15` from [here](http://www.liangzheng.org/Project/project_reid.html), and unzip it in a folder (e.g. `./dataset`)

### Commands
**Run all**

```Shell
./simulation.sh <dataset_path> <feature_path>
```

**DaRe(R)+RE (distance)**

```Shell
python main.py --log_file distance_confidence --confidence_function distance --dataset_path <dataset_path> --feature_path <feature_path> --test_budget
```

**DaRe(R)+RE (margin)**

```Shell
python main.py --log_file margin_confidence --confidence_function margin --dataset_path <dataset_path> --feature_path <feature_path> --test_budget
```

**DaRe(R)+RE (random)**

```Shell
python main.py --log_file random --confidence_function random --dataset_path <dataset_path> --feature_path <feature_path> --test_budget
```

**Plot**

```Shell
python budgeted_stream_plot.py
```

## Notes
* Use `--dump_distance_mat` option to dump the resulted distance matrices and use [this widely used evaluation code](https://github.com/zhunzhong07/person-re-ranking/blob/master/evaluation/Market_1501_evaluation.m) to evaluate the performance (the result should be the same).
* Use `--dump_exit_history` option to dump exit history for each query image, and use these data to generate qualitative results in section 5.4. (We use the history at `budget = 1030203460.27` (corresponding to `q = 0.5`) when using distance confidence function)

## License
MIT
