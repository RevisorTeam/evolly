## Description

Training pipeline presented here is a modified version of the [torchvision's object detection 
example](https://github.com/pytorch/vision/tree/main/references/detection) on COCO 2017 dataset.

COCO 2017 dataset can be [downloaded here](https://cocodataset.org/#download).


## How to launch evolution?

In order to launch evolution proceed over following steps:

1. Download following archives, unpack them and specify paths in the `cfg.py`:
   * 2017 Train images
   * 2017 Val images
   * 2017 Train/Val annotations
2. Run `train.py` to check if the training goes without errors.
3. Run `backbone_search.py`.