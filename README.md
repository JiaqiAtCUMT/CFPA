# Few-shot Object Detection via Context-aware Aggregation for Remote Sensing Images (GRSL 2022)


### Bibtex

```
@ARTICLE{CPFA,  
author={Zhou, Yong and Hu, Han and Zhao, Jiaqi and Zhu, Hancheng and Yao, Rui and Du, Wen-Liang},  journal={IEEE Geoscience and Remote Sensing Letters}, 
title={Few-Shot Object Detection via Context-Aware Aggregation for Remote Sensing Images},   year={2022},  
volume={19},  
number={},  
pages={1-5},  
doi={10.1109/LGRS.2022.3171257}}

```



## Installation

FsDet is built on [Detectron2](https://github.com/facebookresearch/detectron2). But you don't need to build detectron2 seperately as this codebase is self-contained. You can follow the instructions below to install the dependencies and build `FsDet`. FSCE functionalities are implemented as `class`and `.py` scripts in FsDet which therefore requires no extra build efforts. 

**Dependencies**

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.3 
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* Dependencies: ```pip install -r requirements.txt```
* pycocotools: ```pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```
* [fvcore](https://github.com/facebookresearch/fvcore/): ```pip install 'git+https://github.com/facebookresearch/fvcore'``` 
* [OpenCV](https://pypi.org/project/opencv-python/), optional, needed by demo and visualization ```pip install opencv-python```
* GCC >= 4.9

**Build**

```bash
python setup.py build develop  # you might need sudo
```



Note: you may need to rebuild FsDet after reinstalling a different build of PyTorch.




## Code Structure

The code structure follows Detectron2 v0.1.* and fsdet. 

- **configs**: Configuration  files (`YAML`) for train/test jobs. 
- **datasets**: Dataset files (see [Data Preparation](#data-preparation) for more details)
- **fsdet**
  - **checkpoint**: Checkpoint code.
  - **config**: Configuration code and default configurations.
  - **data**: Dataset code.
  - **engine**: Contains training and evaluation loops and hooks.
  - **evaluation**: Evaluation code for different datasets.
  - **layers**: Implementations of different layers used in models.
  - **modeling**: Code for models, including backbones, proposal networks, and prediction heads.
    - The majority of FSCE functionality are implemtended in`modeling/roi_heads/* `, `modeling/contrastive_loss.py`, and  `modeling/utils.py`
    - So one can first make sure  [FsDet v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags) runs smoothly, and then refer to FSCE implementations and configurations. 
  - **solver**: Scheduler and optimizer code.
  - **structures**: Data types, such as bounding boxes and image lists.
  - **utils**: Utility functions.
- **tools**
  - **train_net.py**: Training script.
  - **test_net.py**: Testing script.
  - **ckpt_surgery.py**: Surgery on checkpoints.
  - **run_experiments.py**: Running experiments across many seeds.
  - **aggregate_seeds.py**: Aggregating results from many seeds.



## Train & Inference

### Training

We follow the eaact training procedure of FsDet and we use **random initialization** for novel weights. For a full description of training procedure, see [here](https://github.com/ucbdrive/few-shot-object-detection/blob/master/docs/TRAIN_INST.md).

#### 1. Stage 1: Training base detector.

```
python tools/train_net.py --num-gpus 8 \
        --config-file configs/PASCAL_VOC/base-training/R101_FPN_base_training_split1.yml
```

#### 2. Random initialize  weights for novel classes.

```
python tools/ckpt_surgery.py \
        --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method randinit \
        --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1
```

This step will create a `model_surgery.pth` from` model_final.pth`. 

Don't forget the `--coco` and `--lvis`options when work on the COCO and LVIS datasets, see `ckpt_surgery.py` for all arguments details.

#### 3. Stage 2: Fine-tune for novel data.

```
python tools/train_net.py --num-gpus 8 \
        --config-file configs/PASCAL_VOC/split1/10shot_CL_IoU.yml \
        --opts MODEL.WEIGHTS WEIGHTS_PATH
```

Where `WEIGHTS_PATH` points to the `model_surgery.pth` generated from the previous step. Or you can specify it in the configuration yml. 

#### Evaluation

To evaluate the trained models, run

```angular2html
python tools/test_net.py --num-gpus 8 \
        --config-file configs/PASCAL_VOC/split1/10shot_CL_IoU.yml \
        --eval-only
```

Or you can specify `TEST.EVAL_PERIOD` in the configuation yml to evaluate during training. 






