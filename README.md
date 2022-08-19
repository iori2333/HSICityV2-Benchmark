# Urban Scene Understanding via Hyperspectral Images: Dataset and Benchmark

This is the official code repository for NeurIPS2022 Track **Urban Scene Understanding via Hyperspectral Images: Dataset and Benchmark**.

## Classification

## Segmentation

Our segmentation benchmark is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). We put our benchmarking configs under `segmentation/experiments`.

### Training models

1. You need to read [MMseg Document](https://mmsegmentation.readthedocs.io/) to setup for mmsegmentation framework.
2. Download our HSICityV2 dataset and put it under `data` folder. Run `python tools/convert_datasets/hsicity2.py --root [root to HSICityV2]` to convert dataset.
3. Use `python tools/train.py [config]` (1 GPU) or `bash tools/dist_train.sh [config] [n] `(n GPUs) to train benchmarking models. You can look up the following table to find corresponding config. Note that every config is designed to run with one GPU. If you have more than 1 GPU for training, **you need to change iteration number correspondingly** (e.x 1GPU 160k iters = 2GPUs 80k iters = 4GPUs 40k iters).
4. The config file contains experiment configs, including optimizer, batch size, learning rate, etc. Please refer to [TUTORIAL 1: LEARN ABOUT CONFIGS](https://mmsegmentation.readthedocs.io/en/latest/tutorials/config.html) for more information.

Benchmarks in the paper:
| Model Name           | Config File                                   |
|:--------------------:|:---------------------------------------------:|
| FCN (r50)            | [fcn_r50-d8_0.5x_80k_hsicity2hsi.py](segmentation/experiments/hsicity2-survey/fcn_r50-d8_0.5x_80k_hsicity2hsi.py)        |
| FCN (r101)           | [fcn_r101-d8_0.5x_80k_hsicity2hsi.py](segmentation/experiments/hsicity2-survey/fcn_r101-d8_0.5x_80k_hsicity2hsi.py)          |
| Deeplabv3p (r50)     | [deeplabv3plus_r50-d8_0.5x_80k_hsicity2hsi.py](segmentation/experiments/hsicity2-survey/deeplabv3plus_r50-d8_0.5x_80k_hsicity2hsi.py) |
| HRNet (w48)          | [fcn_hr48_0.5x_80k_bare_hsicity2hsi.py](segmentation/experiments/hsicity2-survey/fcn_hr48_0.5x_80k_bare_hsicity2hsi.py) |
| CCNet (r50)          | [ccnet_r50-d8_0.5x_80k_hsicity2hsi.py](segmentation/experiments/hsicity2-survey/ccnet_r50-d8_0.5x_80k_hsicity2hsi.py) |
| PSPNet (r50)         | [pspnet_r50-d8_0.5x_80k_hsicity2hsi.py](segmentation/experiments/hsicity2-survey/pspnet_r50-d8_0.5x_80k_hsicity2hsi.py) |
| SegFormer (mit-b5)   | [segformer_mit-b5_0.5x_160k_hsicity2hsi.py](segmentation/experiments/hsicity2-survey/segformer_mit-b5_0.5x_160k_hsicity2hsi.py) |
| RTFNet               | [rtfnet_r152-0.5x_80k_hsicity2.py](segmentation/experiments/hsicity2-survey/rtfnet_r152-0.5x_80k_hsicity2.py) |
| FuseNet              | [fusenet_vgg_0.5x_80k_hsicity2.py](segmentation/experiments/hsicity2-survey/fusenet_vgg_0.5x_80k_hsicity2.py) |
| MFNet                | [mfnet_0.5x_80k_hsicity2.py](segmentation/experiments/hsicity2-survey/mfnet_0.5x_80k_hsicity2.py) |
| FCN (r50) RGB        | [fcn_r50-d8_0.5x_80k_hsicity2rgb.py](segmentation/experiments/hsicity2-survey-rgb/fcn_r50-d8_0.5x_80k_hsicity2rgb.py) |
| FCN (r101) RGB       | [fcn_r101-d8_0.5x_80k_hsicity2rgb.py](segmentation/experiments/hsicity2-survey-rgb/fcn_r101-d8_0.5x_80k_hsicity2rgb.py) |
| Deeplabv3p (r50) RGB | [deeplabv3plus_r50-d8_0.5x_80k_hsicity2rgb.py](segmentation/experiments/hsicity2-survey-rgb/deeplabv3plus_r50-d8_0.5x_80k_hsicity2rgb.py) |
| HRNet (w48) RGB      | [fcn_hr48_0.5x_80k_bare_hsicity2rgb.py](segmentation/experiments/hsicity2-survey-rgb/fcn_hr48_0.5x_80k_bare_hsicity2rgb.py) |
| CCNet (r50) RGB      | [ccnet_r50-d8_0.5x_80k_hsicity2rgb.py](segmentation/experiments/hsicity2-survey-rgb/ccnet_r50-d8_0.5x_80k_hsicity2rgb.py) |
| PSPNet (r50) RGB     | [pspnet_r50-d8_0.5x_hsicity2rgb.py](segmentation/experiments/hsicity2-survey-rgb/pspnet_r50-d8_0.5x_hsicity2rgb.py) |
| SegFormer (mit-b5) RGB | [segformer_mit-b5_0.5x_160k_hsicity2rgb.py](segmentation/experiments/hsicity2-survey-rgb/segformer_mit-b5_0.5x_160k_hsicity2rgb.py) |

Additional experiments in Rebuttal:
| Model Name | Config File |
|:----------:|:-----------:|
| FCN-r50 (64->32) | [fcn_r50-d8_0.5x_80k_hsicity2hsi_dconv32.py](segmentation/experiments/hsicity2-survey/fcn_r50-d8_0.5x_80k_hsicity2hsi_dconv32.py) |
| PSPNet-r50 (64->32) | [pspnet_r50-d8_0.5x_80k_hsicity2hsi_dconv32.py](segmentation/experiments/hsicity2-survey/pspnet_r50-d8_0.5x_80k_hsicity2hsi_dconv32.py) |
| FCN-r50 RGB (No pretraining) | [fcn_r50-d8_0.5x_80k_hsicity2rgb_nopretrained.py](segmentation/experiments/hsicity2-survey-rgb/fcn_r50-d8_0.5x_80k_hsicity2rgb_nopretrained.py) |
| PSPNet-r50 RGB (No pretraining) | [pspnet_r50-d8_0.5x_80k_hsicity2rgb_nopretrained.py](segmentation/experiments/hsicity2-survey-rgb/pspnet_r50-d8_0.5x_80k_hsicity2rgb_nopretrained.py) |
| FCN-r50 (ValSet/Coarse) | [fcn_r50-d8_0.5x_80k_hsicity2hsisub_coarse.py](segmentation/experiments/hsicity2-survey/fcn_r50-d8_0.5x_80k_hsicity2hsisub_coarse.py) |
| FCN-r50 (ValSet/Fine) | [fcn_r50-d8_0.5x_80k_hsicity2hsisub_fine.py](segmentation/experiments/hsicity2-survey/fcn_r50-d8_0.5x_80k_hsicity2hsisub_fine.py) |
| PSPNet-r50 (ValSet/Coarse) | [pspnet_r50-d8_0.5x_80k_hsicity2hsisub_coarse.py](segmentation/experiments/hsicity2-survey/pspnet_r50-d8_0.5x_80k_hsicity2hsisub_coarse.py) |
| PSPNet-r50 (ValSet/Fine) | [pspnet_r50-d8_0.5x_80k_hsicity2hsisub_fine.py](segmentation/experiments/hsicity2-survey/pspnet_r50-d8_0.5x_80k_hsicity2hsisub_fine.py) |

### Testing models

Use
```sh
python tools/test.py (config) (trained_model) [--eval_hsi True] [--show-dir dirxxx] [--opacity 1]
```
to test trained models and generate segmentation result. You can refer to [test.py](segmentation\tools\test.py) for more options.