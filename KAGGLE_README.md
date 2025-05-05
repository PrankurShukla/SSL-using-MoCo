# MoCo Pretraining on Kvasir-Instrument Dataset for Kaggle

This repository provides code for Momentum Contrast (MoCo) pretraining on the Kvasir-Instrument dataset, optimized for running on Kaggle.

## Setup in Kaggle

1. Create a new notebook in Kaggle
2. Upload the dataset to Kaggle:
   - Go to "Data" tab
   - Click "New Dataset"
   - Upload the Kvasir-Instrument dataset
   - Make sure the dataset is structured as:
     ```
     kvasir-instrument/
     ├── images/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     └── masks/
         ├── image1.png
         ├── image2.png
         └── ...
     ```

3. In your Kaggle notebook:
   - Add the dataset to your notebook (click "Add data" and select your dataset)
   - Copy the `kaggle_moco.py` file to your notebook
   - Run the cells in the notebook

## Files

- `kaggle_moco.py`: Contains the MoCo model and dataset implementation
- `kaggle_moco_notebook.py`: A script version of the notebook that can be run in Kaggle

## Training Process

The training process follows these steps:

1. Load and preprocess the Kvasir-Instrument dataset
2. Create augmented views of each image for contrastive learning
3. Train the MoCo model using momentum contrastive learning
4. Save checkpoints and the final model

## Model Architecture

The implementation uses:
- ResNet-50 as the backbone
- MoCo v2 with momentum contrastive learning
- Queue size: 4096
- Feature dimension: 128
- Temperature: 0.07
- Momentum: 0.999

## Training Parameters

- Batch size: 128
- Learning rate: 0.03
- Momentum: 0.9
- Weight decay: 1e-4
- Number of epochs: 200

## Using the Pretrained Model

After pretraining, you can use the pretrained encoder for segmentation tasks by:
1. Loading the pretrained weights
2. Adding a segmentation head
3. Fine-tuning on the segmentation task

## Kaggle-Specific Notes

- The code is configured to use Kaggle's GPU accelerators
- Results are saved to `/kaggle/working/results/`
- TensorBoard logs are available for monitoring training progress
- Checkpoints are saved every 10 epochs

## References

- [MoCo v2: Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)
- [Kvasir-Instrument Dataset](https://datasets-server.huggingface.co/datasets/kvasir-instrument) 