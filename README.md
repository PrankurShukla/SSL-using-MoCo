# Surgical Instrument Segmentation with DeepLabV3+ and MoCo Fine-tuning

This repository contains implementations for surgical instrument segmentation using two approaches:
1. DeepLabV3+ with ResNet-50 backbone
2. MoCo (Momentum Contrast) pretraining followed by fine-tuning

## Table of Contents
- [Setup](#setup)
- [Dataset](#dataset)
- [DeepLabV3+ Implementation](#deeplabv3-implementation)
- [MoCo Fine-tuning Implementation](#moco-fine-tuning-implementation)
- [Results and Visualizations](#results-and-visualizations)
- [References](#references)

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/surgical-instrument-segmentation.git
cd surgical-instrument-segmentation
```

## Dataset

The implementations use the Kvasir-Instrument dataset. Download and organize it in the following structure:
```
kvasir_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

## DeepLabV3+ Implementation

### Training
To train the DeepLabV3+ model:

1. Update the dataset path in `deeplabv3_finetune.py`:
```python
DATA_DIR = 'path_to_kvasir_dataset'
```

2. Run the training script:
```bash
python deeplabv3_finetune.py
```

The training script will:
- Train DeepLabV3+ with ResNet-50 backbone
- Use 80% training, 10% validation, and 10% test split
- Implement Dice loss for training
- Calculate IoU and Dice metrics
- Save checkpoints and visualizations
- Train for 50 epochs

### Key Features
- Data augmentation using Albumentations
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping based on validation metrics
- TensorBoard logging
- Visualization of predictions and training curves

## MoCo Fine-tuning Implementation

### Pretraining
To pretrain using MoCo:

1. Update the dataset path in `train_moco.py`:
```python
dataset = KvasirInstrumentMoCoDataset(root_dir='path_to_kvasir_dataset')
```

2. Run the pretraining script:
```bash
python train_moco.py
```

### Fine-tuning
After pretraining, to fine-tune for segmentation:

1. Update the paths in `moco_finetune_comparison_improved.py`:
```python
PRETRAINED_PATH = 'path_to_pretrained_moco_model'
DATA_DIR = 'path_to_kvasir_dataset'
```

2. Run the fine-tuning script:
```bash
python moco_finetune_comparison_improved.py
```

### Key Features
- MoCo v2 implementation with momentum contrastive learning
- Queue size: 4096
- Feature dimension: 128
- Temperature: 0.07
- Momentum: 0.999
- Batch size: 128
- Learning rate: 0.03
- Weight decay: 1e-4
- Number of epochs: 200

## Results and Visualizations

Both implementations include visualization tools:

1. Training curves (loss, IoU, Dice scores)
2. Prediction visualizations
3. Comparison plots between different approaches

To visualize results:
```bash
python visualize_results.py
```

## References

1. DeepLabV3+:
   - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
   - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

2. MoCo:
   - [MoCo v2: Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)
   - [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

3. Dataset:
   - [Kvasir-Instrument Dataset](https://datasets-server.huggingface.co/datasets/kvasir-instrument) 