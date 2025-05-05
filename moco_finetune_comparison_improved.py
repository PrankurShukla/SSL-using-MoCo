# Improved MoCo Pretraining and Segmentation for Kvasir-Instrument Dataset
#
# This notebook implements an improved version of MoCo pretraining followed by surgical instrument segmentation.
# Phase 1: MoCo Pretraining
# Phase 2: Segmentation Fine-tuning

# Install required packages
!pip install -q albumentations
!pip install -q tensorboard
!pip install -q segmentation-models-pytorch
!pip install -q monai

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage
import segmentation_models_pytorch as smp
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.networks.nets import UNet
import random
import warnings
from collections import deque
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# Set up paths
DATA_DIR = '/kaggle/input/kvasir-dataset/kvasir-instrument'
RESULTS_DIR = '/kaggle/working/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

def split_dataset(dataset, test_ratio=0.1, pretrain_ratio=0.5):
    """Split dataset into test, pretraining, and fine-tuning sets"""
    total_size = len(dataset)
    test_size = int(test_ratio * total_size)
    remaining_size = total_size - test_size
    pretrain_size = int(pretrain_ratio * remaining_size)
    
    # Get indices
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Split indices
    test_indices = indices[:test_size]
    pretrain_indices = indices[test_size:test_size + pretrain_size]
    finetune_indices = indices[test_size + pretrain_size:]
    
    return test_indices, pretrain_indices, finetune_indices

class KvasirSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all image files
        image_dir = os.path.join(data_dir, 'images')
        mask_dir = os.path.join(data_dir, 'masks')
        
        if not os.path.exists(image_dir):
            raise RuntimeError(f'Image directory not found: {image_dir}')
        if not os.path.exists(mask_dir):
            raise RuntimeError(f'Mask directory not found: {mask_dir}')
        
        # Get all image files
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.jpg')
        ])
        
        # Verify corresponding masks exist
        valid_pairs = []
        for img_name in self.images:
            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                valid_pairs.append((img_name, mask_name))
        
        if len(valid_pairs) == 0:
            raise RuntimeError(f'No valid image-mask pairs found in {data_dir}')
        
        print(f"Found {len(valid_pairs)} valid image-mask pairs")
        
        self.image_mask_pairs = valid_pairs
    
    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        img_name, mask_name = self.image_mask_pairs[idx]
        img_path = os.path.join(self.data_dir, 'images', img_name)
        mask_path = os.path.join(self.data_dir, 'masks', mask_name)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f'Failed to load image: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f'Failed to load mask: {mask_path}')
        
        # Normalize mask to binary
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Add channel dimension to mask
        mask = mask.unsqueeze(0)
        
        return image, mask, img_name

class MoCoDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.transform = get_moco_augmentation()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, _, _ = self.dataset[original_idx]
        
        # Convert tensor to numpy array and denormalize
        image = image.permute(1, 2, 0).numpy()
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        image = image.astype(np.uint8)
        
        # Apply two different augmentations to the same image
        augmented1 = self.transform(image=image)
        augmented2 = self.transform(image=image)
        
        return augmented1['image'], augmented2['image']

def get_moco_augmentation():
    return A.Compose([
        A.RandomResizedCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.GaussNoise(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_training_augmentation():
    return A.Compose([
        A.RandomResizedCrop(512, 512, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class MoCo(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, device='cuda'):
        super(MoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        
        # Create encoders
        self.encoder_q = models.resnet50(weights=None)
        self.encoder_k = models.resnet50(weights=None)
        
        # Remove the final fc layer
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )
        
        # Initialize key encoder with query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Handle the case where batch_size doesn't divide K
        if ptr + batch_size > self.K:
            # Fill the remaining space
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            # Start from the beginning
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        else:
            # Normal case
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
        
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # Compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        
        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        
        # Compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # apply temperature
        logits /= self.T
        
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logits, labels

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DeepLabV3PlusMoCo(nn.Module):
    def __init__(self, moco_path):
        super().__init__()
        
        # Load MoCo state dict
        moco_state = torch.load(moco_path)
        
        # Create DeepLabV3+ model
        self.model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=1)
        
        # Get the state dict of MoCo's encoder_q
        moco_dict = moco_state['model_state_dict']
        encoder_dict = {k.replace('encoder_q.', ''): v for k, v in moco_dict.items() 
                       if k.startswith('encoder_q.') and not k.startswith('encoder_q.fc')}
        
        # Load MoCo weights into backbone
        missing_keys, unexpected_keys = self.model.backbone.load_state_dict(encoder_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    
    def forward(self, x):
        output = self.model(x)
        return output

def dice_loss(pred, target):
    smooth = 1.0
    # Handle model output dictionary
    if isinstance(pred, dict):
        pred = pred['out']
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def train_one_epoch(model, loader, criterion, optimizer, device, model_type):
    model.train()
    total_loss = 0
    
    with tqdm(loader) as pbar:
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device, writer=None, epoch=None, save_dir=None, model_type=None):
    model.eval()
    total_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate Dice score
            if model_type == 'unet':
                pred = torch.sigmoid(outputs)
            else:  # deeplabv3plus
                pred = torch.sigmoid(outputs['out'])
            
            pred = (pred > 0.5).float()
            dice = (2.0 * (pred * masks).sum()) / (pred.sum() + masks.sum() + 1e-8)
            dice_scores.append(dice.item())
            
            total_loss += loss.item()
    
    return total_loss / len(loader), np.mean(dice_scores)

def test_model(model, test_loader, device, save_dir=None, model_type=None):
    """Evaluate model on test set and calculate metrics"""
    model.eval()
    total_iou = 0
    total_dice = 0
    all_ious = []
    all_dices = []
    
    # Create directory for saving test results
    if save_dir:
        os.makedirs(os.path.join(save_dir, f'{model_type}_test_results'), exist_ok=True)
    
    print(f"\nEvaluating {model_type} on test set...")
    with torch.no_grad():
        for i, (images, masks, filenames) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            if model_type == 'unet':
                preds = torch.sigmoid(outputs)
            else:  # deeplabv3plus
                preds = torch.sigmoid(outputs['out'])
            
            pred_masks = (preds > 0.5).float()
            
            # Calculate metrics for each image in batch
            for j in range(len(images)):
                pred_mask = pred_masks[j, 0].cpu().numpy()
                gt_mask = masks[j, 0].cpu().numpy()
                
                # Calculate IoU
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                iou = intersection / (union + 1e-8)
                
                # Calculate Dice
                dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)
                
                total_iou += iou
                total_dice += dice
                all_ious.append(iou)
                all_dices.append(dice)
                
                # Save visualization if requested
                if save_dir:
                    plt.figure(figsize=(15, 5))
                    
                    # Original image
                    plt.subplot(131)
                    img = images[j].cpu().numpy().transpose(1, 2, 0)
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    plt.imshow(img)
                    plt.title('Original Image')
                    plt.axis('off')
                    
                    # Ground truth
                    plt.subplot(132)
                    plt.imshow(gt_mask, cmap='gray')
                    plt.title('Ground Truth')
                    plt.axis('off')
                    
                    # Prediction
                    plt.subplot(133)
                    plt.imshow(pred_mask, cmap='gray')
                    plt.title(f'Prediction (IoU: {iou:.3f}, Dice: {dice:.3f})')
                    plt.axis('off')
                    
                    plt.suptitle(f'{model_type.upper()} - Test Sample: {filenames[j]}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'{model_type}_test_results', f'{model_type}_test_{filenames[j][:-4]}.png'))
                    plt.close()
    
    # Calculate average metrics
    avg_iou = total_iou / len(test_loader.dataset)
    avg_dice = total_dice / len(test_loader.dataset)
    
    # Calculate standard deviation
    std_iou = np.std(all_ious)
    std_dice = np.std(all_dices)
    
    # Calculate min and max metrics
    min_iou = min(all_ious)
    max_iou = max(all_ious)
    min_dice = min(all_dices)
    max_dice = max(all_dices)
    
    # Print results
    print(f"\n{model_type.upper()} Test Results:")
    print(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"IoU Range: {min_iou:.4f} - {max_iou:.4f}")
    print(f"Dice Range: {min_dice:.4f} - {max_dice:.4f}")
    
    # Save metrics to file
    if save_dir:
        with open(os.path.join(save_dir, f'{model_type}_test_metrics.txt'), 'w') as f:
            f.write(f"{model_type.upper()} Test Results:\n")
            f.write(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}\n")
            f.write(f"Average Dice: {avg_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"IoU Range: {min_iou:.4f} - {max_iou:.4f}\n")
            f.write(f"Dice Range: {min_dice:.4f} - {max_dice:.4f}\n")
            f.write(f"Number of test samples: {len(test_loader.dataset)}\n")
    
    # Create a summary visualization of all test results
    if save_dir:
        plt.figure(figsize=(10, 6))
        plt.hist(all_ious, bins=20, alpha=0.7, label='IoU')
        plt.hist(all_dices, bins=20, alpha=0.7, label='Dice')
        plt.axvline(avg_iou, color='blue', linestyle='dashed', linewidth=1, label=f'Avg IoU: {avg_iou:.3f}')
        plt.axvline(avg_dice, color='orange', linestyle='dashed', linewidth=1, label=f'Avg Dice: {avg_dice:.3f}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of IoU and Dice Scores on Test Set - {model_type.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'{model_type}_test_results', f'{model_type}_score_distribution.png'))
        plt.close()
    
    return avg_iou, avg_dice, all_ious, all_dices

def train_model(model_type, train_loader, val_loader, device, num_epochs=50, save_dir=None, moco_path=None):
    """Train a model and return the best model and metrics"""
    # Create model
    if model_type == 'unet':
        model = UNet(n_channels=3, n_classes=1).to(device)
    else:  # deeplabv3plus
        model = DeepLabV3PlusMoCo(moco_path).to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create criterion
    criterion = dice_loss
    
    # Initialize tensorboard
    writer = SummaryWriter(os.path.join(save_dir, f'{model_type}_logs'))
    
    # Training loop
    best_dice = 0.0
    best_model = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, model_type)
        
        # Validate
        val_loss, dice_score = validate(
            model, 
            val_loader, 
            criterion, 
            device,
            writer=writer,
            epoch=epoch,
            save_dir=save_dir,
            model_type=model_type
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar(f'{model_type}/Loss/train', train_loss, epoch)
        writer.add_scalar(f'{model_type}/Loss/val', val_loss, epoch)
        writer.add_scalar(f'{model_type}/Dice/val', dice_score, epoch)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Dice Score: {dice_score:.4f}")
        
        # Save best model
        if dice_score > best_dice:
            best_dice = dice_score
            best_model = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(save_dir, f'best_{model_type}_model.pth'))
            print(f"Saved best model with Dice score: {best_dice:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(save_dir, f'{model_type}_checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print(f"{model_type.upper()} training completed!")
    
    # Load best model
    model.load_state_dict(best_model)
    return model, best_dice

def pretrain_moco(model, pretrain_loader, device, num_epochs=100, save_dir=None):
    """Pretrain model using MoCo framework"""
    print("\nStarting MoCo pretraining...")
    
    # Initialize tensorboard
    writer = SummaryWriter(os.path.join(save_dir, 'moco_pretrain_logs'))
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(pretrain_loader) as pbar:
            for view1, view2 in pbar:
                view1 = view1.to(device)
                view2 = view2.to(device)
                
                # Forward pass
                logits, labels = model(view1, view2)
                loss = criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', total_loss / len(pretrain_loader), epoch)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'moco_checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(save_dir, 'final_moco_model.pth'))
    
    writer.close()
    print("MoCo pretraining completed!")

def main():
    # Set random seed
    seed_everything(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_DIR, f'comparison_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create full dataset
    full_dataset = KvasirSegmentationDataset(DATA_DIR, transform=get_training_augmentation())
    
    # Split dataset
    test_indices, pretrain_indices, finetune_indices = split_dataset(full_dataset)
    
    # Create datasets
    test_dataset = Subset(full_dataset, test_indices)
    pretrain_dataset = MoCoDataset(full_dataset, pretrain_indices)
    finetune_dataset = Subset(full_dataset, finetune_indices)
    
    # Split finetune dataset into train and val
    finetune_size = len(finetune_dataset)
    val_size = int(0.2 * finetune_size)
    train_size = finetune_size - val_size
    
    finetune_indices = list(range(finetune_size))
    random.shuffle(finetune_indices)
    
    train_indices = finetune_indices[:train_size]
    val_indices = finetune_indices[train_size:]
    
    train_dataset = Subset(finetune_dataset, train_indices)
    val_dataset = Subset(finetune_dataset, val_indices)
    
    print(f"Dataset split:")
    print(f"- Test: {len(test_dataset)} samples (10%)")
    print(f"- Pretraining: {len(pretrain_dataset)} samples (45%)")
    print(f"- Fine-tuning: {len(finetune_dataset)} samples (45%)")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Val: {len(val_dataset)} samples")
    
    # Create data loaders
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize MoCo model
    moco_model = MoCo(device=device)
    moco_model = moco_model.to(device)
    
    # Pretrain with MoCo
    pretrain_moco(moco_model, pretrain_loader, device, save_dir=results_dir)
    
    # Fine-tune DeepLabV3+
    print("\nFine-tuning DeepLabV3+...")
    moco_path = os.path.join(results_dir, 'final_moco_model.pth')
    deeplab_model = DeepLabV3PlusMoCo(moco_path).to(device)
    deeplab_model, best_dice_deeplab = train_model(
        'deeplabv3plus',
        train_loader,
        val_loader,
        device,
        save_dir=os.path.join(results_dir, 'deeplabv3plus'),
        moco_path=moco_path
    )
    
    # Fine-tune UNet
    print("\nFine-tuning UNet...")
    unet_model = UNet(n_channels=3, n_classes=1).to(device)
    unet_model, best_dice_unet = train_model(
        'unet',
        train_loader,
        val_loader,
        device,
        save_dir=os.path.join(results_dir, 'unet')
    )
    
    # Evaluate both models on test set
    print("\nEvaluating models on test set...")
    
    # Evaluate DeepLabV3+
    test_iou_deeplab, test_dice_deeplab, _, _ = test_model(
        deeplab_model,
        test_loader,
        device,
        save_dir=os.path.join(results_dir, 'deeplabv3plus_test_results')
    )
    
    # Evaluate UNet
    test_iou_unet, test_dice_unet, _, _ = test_model(
        unet_model,
        test_loader,
        device,
        save_dir=os.path.join(results_dir, 'unet_test_results')
    )
    
    # Save final metrics
    with open(os.path.join(results_dir, 'final_metrics.txt'), 'w') as f:
        f.write("Methodology Summary:\n")
        f.write("1. Split dataset into 10% test, 45% pretraining, 45% fine-tuning\n")
        f.write("2. Pretrain with MoCo on 45% of data\n")
        f.write("3. Fine-tune both models on remaining 45% of data\n")
        f.write("4. Evaluate on 10% test set\n\n")
        
        f.write("DeepLabV3+ Results:\n")
        f.write(f"Best validation Dice: {best_dice_deeplab:.4f}\n")
        f.write(f"Test IoU: {test_iou_deeplab:.4f}\n")
        f.write(f"Test Dice: {test_dice_deeplab:.4f}\n\n")
        
        f.write("UNet Results:\n")
        f.write(f"Best validation Dice: {best_dice_unet:.4f}\n")
        f.write(f"Test IoU: {test_iou_unet:.4f}\n")
        f.write(f"Test Dice: {test_dice_unet:.4f}\n")
    
    print("\nTraining and evaluation completed!")
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main() 