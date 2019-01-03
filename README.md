# Resource Aware Person Re-identification across Multiple Resolutions

This repository contains the code for paper
"[Resource Aware Person Re-identification across Multiple Resolutions](https://arxiv.org/abs/1805.08805)"
(CVPR 2018).

## Citation
```
@inproceedings{wang2018resource,
  title={Resource Aware Person Re-identification across Multiple Resolutions},
  author={Wang, Yan and Wang, Lequn and You, Yurong and Zou, Xu and Chen, Vincent and Li, Serena and Huang, Gao and Hariharan, Bharath and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8042--8051},
  year={2018}
}
```

## Usage

### Dependencies
- [Python3.6](https://www.python.org/downloads/)
- [PyTorch(0.2.0)](http://pytorch.org)
- [torchvision(0.2.0)](http://pytorch.org)
- [Market1501 dataset](http://www.liangzheng.org/Project/project_reid.html)
- [MARS dataset](http://www.liangzheng.com.cn/Project/project_mars.html)
- [CUHK03 dataset](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP)
- [DukeMTMC-reID dataset](https://github.com/layumi/DukeMTMC-reID_evaluation)

### Usage
#### Training Dataset Preprocessing
Use the following command to preprocess the person re-id dataset.

```bash
python create_market_dataset.py --path <root_path_of_dataset>
```

#### Train
Use the following command to set up the training.

```bash
./train.sh <nettype> <GPU> <train_dataset_path> <checkpoint_name>
```
where `<nettype>` can be either `dare_R` or `dare_D`

#### Extract Features
Use the following command to load a trained model to generate features for each image (in `.mat` format).

```bash
./extract_features.sh <nettype> <GPU> <dataset_path> <dataset> <checkpoint_name> <feature_path> <gen_stage_features>
```
where `<nettype>` can be either `dare_R` or `dare_D`, `<dataset>` can be one of `[MARS, Market1501, Duke, CUHK03]`, `<feature_path>` is the path to store extracted features. Toggle `<gen_stage_features>` to `Ture` to extract features from each stage.

#### Evaluation

Use [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking) and [MARS-evaluation](https://github.com/liangzheng06/MARS-evaluation) official evaluation codes to evaluate the extracted features.
Note we use `mean` rather than `max` to aggregate the image feature vectors for video sequences.

#### Resource-aware Person Re-ID Simulation
Use the following command to run simulations under resource-aware person re-ID scenarios. See [here](budgeted_stream/README.md) for more information.

```bash
./budgeted_stream/simulation.sh <dataset_path> <feature_path>
```

## Pretrained Model

We provide several pretrained models listed below:

- [Market1501 Res50](https://drive.google.com/file/d/1u4HD-9vlyfpc9sKEqUTcsm1-whR3bjTo/view?usp=sharing)
- [Market1501 Dense201](https://drive.google.com/file/d/1nJ_GYXbkFI26BCkcEmuCIsJbYqzEk6YL/view?usp=sharing)
- [MARS Res50](https://drive.google.com/file/d/1Adv3dbL_2PWURWYA5TA1HErdVu2DVOGv/view?usp=sharing)
- [MARS Dense201](https://drive.google.com/file/d/1_WS38dhRNp8C9t0itEdI2LI6A4rqYKJ_/view?usp=sharing)
- [CUHK Detected Res50](https://drive.google.com/file/d/12qrsilTGQ9X9MhFwR2g3AHHDT7UsKnIn/view?usp=sharing)
- [CUHK Detected Dense201](https://drive.google.com/file/d/1EEHhAff28_L2u-G14jg0MHbO_ManQnfD/view?usp=sharing)
- [CUHK Labeled Res50](https://drive.google.com/file/d/1AJY2u8PMWtTkLoRvOEcnSF_QR3Cx9gnX/view?usp=sharing)
- [CUHK Labeled Dense201](https://drive.google.com/file/d/1IsVEYc2AV2cGovt015cQ3WcL48U-tFik/view?usp=sharing)
- [Duke Res50](https://drive.google.com/file/d/1B1BR9p6K-wW1oOkmDQZPfiyj2l4zcdc9/view?usp=sharing)
- [Duke Dense201](https://drive.google.com/file/d/1BwfjlMk3K7sgBPcBs6gzCciBC6X8Q9hL/view?usp=sharing)


## License
MIT

