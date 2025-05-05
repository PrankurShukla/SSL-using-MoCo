# %% [markdown]
# # MoCo Pretraining on Kvasir-Instrument Dataset
# 
# This notebook implements Momentum Contrast (MoCo) pretraining for the Kvasir-Instrument dataset.

# %%
# Install required packages
!pip install albumentations tensorboard

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import torchvision.models as models

# Set up Kaggle paths
DATA_DIR = "/kaggle/input/kvasir-dataset/kvasir-instrument"  # Modified for Kaggle
RESULTS_DIR = "/kaggle/working/results"       # Modified for Kaggle
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %%
# Import the MoCo model and dataset from the kaggle_moco.py file
from kaggle_moco import MoCo, KvasirInstrumentMoCoDataset

# %%
# Check if the dataset exists and print some information
if os.path.exists(DATA_DIR):
    print(f"Dataset found at {DATA_DIR}")
    
    # Check image directory
    image_dir = os.path.join(DATA_DIR, 'images')
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        print(f"Found {len(image_files)} images")
        
        # Display a sample image
        if len(image_files) > 0:
            sample_img_path = os.path.join(image_dir, image_files[0])
            sample_img = cv2.imread(sample_img_path)
            sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(sample_img)
            plt.title(f"Sample image: {image_files[0]}")
            plt.axis('off')
            plt.show()
    else:
        print(f"Image directory not found at {image_dir}")
else:
    print(f"Dataset not found at {DATA_DIR}")

# %%
# Training parameters
batch_size = 128
epochs = 200
lr = 0.03
momentum = 0.9
weight_decay = 1e-4

# MoCo parameters
dim = 128
K = 4096
m = 0.999
T = 0.07

# Initialize tensorboard
writer = SummaryWriter(log_dir=os.path.join(RESULTS_DIR, 'logs'))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create dataset and dataloader
dataset = KvasirInstrumentMoCoDataset(root_dir=DATA_DIR)
print(f"Dataset size: {len(dataset)}")

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# Create model
model = MoCo(
    dim=dim,
    K=K,
    m=m,
    T=T,
    arch='resnet50'
).to(device)

# Create optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=momentum,
    weight_decay=weight_decay
)

# Create loss function
criterion = nn.CrossEntropyLoss().to(device)

# %%
# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
    for images, _ in progress_bar:
        # Get the two augmented views
        im_q = images[0].to(device)
        im_k = images[1].to(device)
        
        # Forward pass
        output, target = model(im_q, im_k)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
    
    # Log metrics
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(RESULTS_DIR, f'checkpoint_epoch_{epoch+1}.pth')
        )
        print(f"Saved checkpoint at epoch {epoch+1}")

writer.close()
print("Training completed!")

# %%
# Save the final model
final_checkpoint = {
    'epoch': epochs,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(
    final_checkpoint,
    os.path.join(RESULTS_DIR, 'final_model.pth')
)
print("Final model saved!") 