import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
from datetime import datetime
import tensorflow as tf
import torch.nn.functional as F

# Set paths
DATA_DIR = '/kaggle/input/kvasir-dataset/kvasir-instrument'
RESULTS_DIR = '/kaggle/working/results'
MOCO_CHECKPOINT = os.path.join(RESULTS_DIR, 'final_moco_model.pth')

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

# UNet Model Definition
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

class UNetMoCo(nn.Module):
    def __init__(self, moco_path):
        super().__init__()
        
        # Load MoCo state dict
        moco_state = torch.load(moco_path, weights_only=True)
        
        # Create UNet model
        self.model = UNet(n_channels=3, n_classes=1)
        
        # Print a sample of the initial encoder weights
        print("\nInitial UNet encoder weights (first layer):")
        for name, param in self.model.inc.named_parameters():
            if 'double_conv.0.weight' in name:
                print(f"{name}: {param[0, 0, 0, 0].item()}")
                break
        
        # Get the state dict of MoCo's encoder_q
        moco_dict = moco_state['model_state_dict']
        encoder_dict = {k.replace('encoder_q.', ''): v for k, v in moco_dict.items() 
                       if k.startswith('encoder_q.') and not k.startswith('encoder_q.fc')}
        
        # Map MoCo weights to UNet encoder
        # This is a simplified mapping - in practice, you might need a more complex mapping
        # depending on the exact architecture of your MoCo encoder
        mapped_dict = {}
        for k, v in encoder_dict.items():
            if 'conv1' in k:
                mapped_dict['inc.double_conv.0.weight'] = v
            elif 'bn1' in k:
                mapped_dict['inc.double_conv.1.weight'] = v
                mapped_dict['inc.double_conv.1.bias'] = encoder_dict[k.replace('weight', 'bias')]
                mapped_dict['inc.double_conv.1.running_mean'] = encoder_dict[k.replace('weight', 'running_mean')]
                mapped_dict['inc.double_conv.1.running_var'] = encoder_dict[k.replace('weight', 'running_var')]
            elif 'layer1.0' in k:
                mapped_dict['down1.1.double_conv.0.weight'] = v
            elif 'layer1.1' in k:
                mapped_dict['down1.1.double_conv.3.weight'] = v
        
        # Load mapped MoCo weights into UNet encoder
        missing_keys, unexpected_keys = self.model.load_state_dict(mapped_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
        # Print a sample of the loaded MoCo weights
        print("\nLoaded MoCo weights (first layer):")
        for name, param in self.model.inc.named_parameters():
            if 'double_conv.0.weight' in name:
                print(f"{name}: {param[0, 0, 0, 0].item()}")
                break
    
    def forward(self, x):
        return self.model(x)

# DeepLabV3+ Model Definition
class DeepLabV3PlusMoCo(nn.Module):
    def __init__(self, moco_path):
        super().__init__()
        
        # Load MoCo state dict
        moco_state = torch.load(moco_path, weights_only=True)
        
        # Create DeepLabV3+ model
        self.model = deeplabv3_resnet50(
            weights=None,
            num_classes=1,
            aux_loss=None
        )
        
        # Print a sample of the initial backbone weights
        print("\nInitial DeepLabV3+ backbone weights (first layer):")
        for name, param in self.model.backbone.named_parameters():
            if 'conv1.weight' in name:
                print(f"{name}: {param[0, 0, 0, 0].item()}")
                break
        
        # Get the state dict of MoCo's encoder_q
        moco_dict = moco_state['model_state_dict']
        encoder_dict = {k.replace('encoder_q.', ''): v for k, v in moco_dict.items() 
                       if k.startswith('encoder_q.') and not k.startswith('encoder_q.fc')}
        
        # Load MoCo weights into backbone
        missing_keys, unexpected_keys = self.model.backbone.load_state_dict(encoder_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
        # Print a sample of the loaded MoCo weights
        print("\nLoaded MoCo weights (first layer):")
        for name, param in self.model.backbone.named_parameters():
            if 'conv1.weight' in name:
                print(f"{name}: {param[0, 0, 0, 0].item()}")
                break
    
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
    
    # Store some examples for visualization
    vis_images = []
    vis_masks = []
    vis_preds = []
    
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(loader):
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
            
            # Store first batch for visualization
            if i == 0:
                vis_images.extend([img for img in images[:4]])
                vis_masks.extend([mask for mask in masks[:4]])
                vis_preds.extend([p for p in pred[:4]])
    
    # Log images to tensorboard
    if writer is not None and epoch is not None:
        # Create visualization grid
        vis_images = torch.stack(vis_images)
        vis_masks = torch.stack(vis_masks)
        vis_preds = torch.stack(vis_preds)
        
        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        vis_images = vis_images * std + mean
        
        # Create grids
        image_grid = make_grid(vis_images, nrow=4, normalize=True)
        mask_grid = make_grid(vis_masks, nrow=4)
        pred_grid = make_grid(vis_preds, nrow=4)
        
        # Log to tensorboard
        writer.add_image(f'{model_type}/Images/Input', image_grid, epoch)
        writer.add_image(f'{model_type}/Images/GroundTruth', mask_grid, epoch)
        writer.add_image(f'{model_type}/Images/Prediction', pred_grid, epoch)
    
    # Save detailed visualizations
    if save_dir is not None:
        os.makedirs(os.path.join(save_dir, f'{model_type}_visualizations'), exist_ok=True)
        visualize_predictions(
            model, 
            loader.dataset, 
            device, 
            num_samples=4,
            save_dir=os.path.join(save_dir, f'{model_type}_visualizations'),
            epoch=epoch,
            model_type=model_type
        )
    
    return total_loss / len(loader), np.mean(dice_scores)

def visualize_predictions(model, dataset, device, num_samples=4, save_dir=None, epoch=None, model_type=None):
    """Visualize predictions for a few samples"""
    model.eval()
    
    # Create a figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    # Get random samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            # Get sample
            image, mask, _ = dataset[sample_idx]
            
            # Add batch dimension
            image = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image)
            if model_type == 'unet':
                pred = torch.sigmoid(output).cpu()
            else:  # deeplabv3plus
                pred = torch.sigmoid(output['out']).cpu()
            
            pred_mask = (pred > 0.5).float()
            
            # Convert tensors to numpy arrays
            image = image.cpu().squeeze().permute(1,2,0).numpy()
            image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            image = image.astype(np.uint8)
            
            mask = mask.squeeze().numpy()
            pred_mask = pred_mask.squeeze().numpy()
            
            # Plot
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title('Input Image')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(mask, cmap='gray')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred_mask, cmap='gray')
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        epoch_str = f'_epoch_{epoch}' if epoch is not None else ''
        plt.savefig(os.path.join(save_dir, f'{model_type}_predictions{epoch_str}.png'))
        plt.close()
    else:
        plt.show()

def calculate_metrics(pred_mask, gt_mask):
    """Calculate IoU and Dice score for binary segmentation"""
    # Convert to binary masks
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Calculate IoU
    iou = intersection / (union + 1e-8)
    
    # Calculate Dice score
    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)
    
    return iou, dice

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
                
                iou, dice = calculate_metrics(pred_mask, gt_mask)
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

def create_confusion_matrix(model, test_loader, device, save_dir=None, model_type=None):
    """Create confusion matrix visualization for the test set"""
    model.eval()
    all_preds = []
    all_targets = []
    
    print(f"\nCreating confusion matrix for {model_type}...")
    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            if model_type == 'unet':
                preds = torch.sigmoid(outputs)
            else:  # deeplabv3plus
                preds = torch.sigmoid(outputs['out'])
            
            pred_masks = (preds > 0.5).float()
            
            # Flatten predictions and targets
            pred_masks = pred_masks.cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()
            
            all_preds.extend(pred_masks)
            all_targets.extend(masks)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Background', 'Polyp'],
                yticklabels=['Background', 'Polyp'])
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    
    # Add metrics text
    plt.text(0.5, -0.3, f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}',
             horizontalalignment='center', transform=plt.gca().transAxes)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{model_type}_test_results', f'{model_type}_confusion_matrix.png'), bbox_inches='tight')
        plt.close()
        
        # Save metrics to file
        with open(os.path.join(save_dir, f'{model_type}_test_metrics.txt'), 'a') as f:
            f.write(f"\nConfusion Matrix Metrics:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
    else:
        plt.show()
    
    return cm, accuracy, precision, recall, f1

def evaluate_training_data(model, train_loader, device, save_dir=None):
    """Evaluate model on training data and calculate metrics"""
    model.eval()
    total_iou = 0
    total_dice = 0
    all_ious = []
    all_dices = []
    
    # Create directory for saving training results
    if save_dir:
        os.makedirs(os.path.join(save_dir, 'training_results'), exist_ok=True)
    
    print("\nEvaluating on training data...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)['out']
            preds = torch.sigmoid(outputs)
            pred_masks = (preds > 0.5).float()
            
            # Calculate metrics for each image in batch
            for j in range(len(images)):
                pred_mask = pred_masks[j, 0].cpu().numpy()
                gt_mask = masks[j, 0].cpu().numpy()
                
                iou, dice = calculate_metrics(pred_mask, gt_mask)
                total_iou += iou
                total_dice += dice
                all_ious.append(iou)
                all_dices.append(dice)
                
                # Save visualization if requested
                if save_dir and i == 0:  # Only save first batch for visualization
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
                    
                    plt.suptitle('Training Sample')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'training_results', f'training_sample_{j}.png'))
                    plt.close()
    
    # Calculate average metrics
    avg_iou = total_iou / len(train_loader.dataset)
    avg_dice = total_dice / len(train_loader.dataset)
    
    # Calculate standard deviation
    std_iou = np.std(all_ious)
    std_dice = np.std(all_dices)
    
    # Print results
    print(f"\nTraining Data Results:")
    print(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f} ± {std_dice:.4f}")
    
    # Save metrics to file
    if save_dir:
        with open(os.path.join(save_dir, 'training_metrics.txt'), 'w') as f:
            f.write(f"Training Data Results:\n")
            f.write(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}\n")
            f.write(f"Average Dice: {avg_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"Number of training samples: {len(train_loader.dataset)}\n")
    
    # Create distribution plots
    if save_dir:
        plt.figure(figsize=(12, 5))
        
        # IoU distribution
        plt.subplot(121)
        plt.hist(all_ious, bins=20, alpha=0.7)
        plt.axvline(avg_iou, color='r', linestyle='dashed', linewidth=1)
        plt.title('IoU Score Distribution')
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        
        # Dice distribution
        plt.subplot(122)
        plt.hist(all_dices, bins=20, alpha=0.7)
        plt.axvline(avg_dice, color='r', linestyle='dashed', linewidth=1)
        plt.title('Dice Score Distribution')
        plt.xlabel('Dice Score')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_results', 'score_distributions.png'))
        plt.close()
    
    return avg_iou, avg_dice, all_ious, all_dices

def train_model(model_type, train_loader, val_loader, device, num_epochs=50, save_dir=None):
    """Train a model and return the best model and metrics"""
    # Create model
    if model_type == 'unet':
        model = UNet(n_channels=3, n_classes=1).to(device)
    else:  # deeplabv3plus
        model = DeepLabV3PlusMoCo(MOCO_CHECKPOINT).to(device)
    
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
        
        # Validate with visualizations
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

class MoCoDataset(Dataset):
    """Dataset for MoCo pretraining that returns two augmented views of the same image"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.transform = get_moco_augmentation()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, _, _ = self.dataset[original_idx]
        
        # Apply two different augmentations to the same image
        augmented1 = self.transform(image=image)
        augmented2 = self.transform(image=image)
        
        return augmented1['image'], augmented2['image']

def get_moco_augmentation():
    """Get augmentation pipeline for MoCo pretraining"""
    return A.Compose([
        A.RandomResizedCrop(512, 512, scale=(0.2, 1.0)),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class MoCo(nn.Module):
    """Momentum Contrast for Unsupervised Visual Representation Learning"""
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, device='cuda'):
        super(MoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        
        # Create encoders
        self.encoder_q = deeplabv3_resnet50(weights=None, num_classes=dim)
        self.encoder_k = deeplabv3_resnet50(weights=None, num_classes=dim)
        
        # Remove the classifier head
        self.encoder_q.classifier = nn.Identity()
        self.encoder_k.classifier = nn.Identity()
        
        # Add projection head
        self.projection_q = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )
        self.projection_k = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )
        
        # Initialize key encoder with query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
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
        
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        
        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        
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
        q = self.encoder_q(im_q)['out']  # queries: NxC
        q = self.projection_q(q.mean(dim=[2, 3]))  # Global average pooling
        q = nn.functional.normalize(q, dim=1)
        
        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            
            k = self.encoder_k(im_k)['out']  # keys: NxC
            k = self.projection_k(k.mean(dim=[2, 3]))  # Global average pooling
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

def plot_training_curves(model_type, save_dir):
    """Plot training and validation loss curves"""
    # Read tensorboard logs
    log_dir = os.path.join(save_dir, f'{model_type}_logs')
    train_losses = []
    val_losses = []
    val_dice = []
    
    # Read events file
    for event in tf.compat.v1.train.summary_iterator(os.path.join(log_dir, 'events.out.tfevents.*')):
        for value in event.summary.value:
            if value.tag == f'{model_type}/Loss/train':
                train_losses.append(value.simple_value)
            elif value.tag == f'{model_type}/Loss/val':
                val_losses.append(value.simple_value)
            elif value.tag == f'{model_type}/Dice/val':
                val_dice.append(value.simple_value)
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(121)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.upper()} Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Dice score
    plt.subplot(122)
    plt.plot(val_dice, label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title(f'{model_type.upper()} Validation Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}_training_curves.png'))
    plt.close()

def visualize_all_test_results(model, test_loader, device, save_dir, model_type):
    """Visualize predictions for all test samples"""
    model.eval()
    os.makedirs(os.path.join(save_dir, 'all_test_results'), exist_ok=True)
    
    print(f"\nVisualizing all test results for {model_type}...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            if model_type == 'unet':
                preds = torch.sigmoid(outputs)
            else:  # deeplabv3plus
                preds = torch.sigmoid(outputs['out'])
            
            pred_masks = (preds > 0.5).float()
            
            # Visualize each image in batch
            for j in range(len(images)):
                # Create figure
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
                plt.imshow(masks[j, 0].cpu().numpy(), cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')
                
                # Prediction
                plt.subplot(133)
                plt.imshow(pred_masks[j, 0].cpu().numpy(), cmap='gray')
                plt.title('Prediction')
                plt.axis('off')
                
                plt.suptitle(f'{model_type.upper()} - Test Sample {i*test_loader.batch_size + j}')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'all_test_results', 
                                       f'{model_type}_test_{i*test_loader.batch_size + j}.png'))
                plt.close()

def main():
    # Set random seed
    set_seed(42)
    
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
    deeplab_model = DeepLabV3PlusMoCo(os.path.join(results_dir, 'final_moco_model.pth')).to(device)
    deeplab_model, best_dice_deeplab = train_model(
        'deeplabv3plus',
        train_loader,
        val_loader,
        device,
        save_dir=os.path.join(results_dir, 'deeplabv3plus')
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

    # After training both models, plot training curves
    print("\nPlotting training curves...")
    plot_training_curves('deeplabv3plus', os.path.join(results_dir, 'deeplabv3plus'))
    plot_training_curves('unet', os.path.join(results_dir, 'unet'))
    
    # Visualize all test results for both models
    print("\nVisualizing all test results...")
    visualize_all_test_results(
        deeplab_model,
        test_loader,
        device,
        os.path.join(results_dir, 'deeplabv3plus_test_results'),
        'deeplabv3plus'
    )
    
    visualize_all_test_results(
        unet_model,
        test_loader,
        device,
        os.path.join(results_dir, 'unet_test_results'),
        'unet'
    )
    
    # Create a summary visualization of test results
    print("\nCreating summary visualization...")
    plt.figure(figsize=(20, 10))
    
    # Get a few samples from each model's predictions
    num_samples = min(5, len(test_loader.dataset))
    sample_indices = random.sample(range(len(test_loader.dataset)), num_samples)
    
    for i, idx in enumerate(sample_indices):
        # Get sample
        image, mask = test_loader.dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        # Get DeepLabV3+ prediction
        with torch.no_grad():
            deeplab_output = deeplab_model(image)['out']
            deeplab_pred = torch.sigmoid(deeplab_output)
            deeplab_mask = (deeplab_pred > 0.5).float()
            
            unet_output = unet_model(image)
            unet_pred = torch.sigmoid(unet_output)
            unet_mask = (unet_pred > 0.5).float()
        
        # Original image
        plt.subplot(num_samples, 4, i*4 + 1)
        img = image[0].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(num_samples, 4, i*4 + 2)
        plt.imshow(mask[0].cpu().numpy(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # DeepLabV3+ prediction
        plt.subplot(num_samples, 4, i*4 + 3)
        plt.imshow(deeplab_mask[0, 0].cpu().numpy(), cmap='gray')
        plt.title('DeepLabV3+ Prediction')
        plt.axis('off')
        
        # UNet prediction
        plt.subplot(num_samples, 4, i*4 + 4)
        plt.imshow(unet_mask[0, 0].cpu().numpy(), cmap='gray')
        plt.title('UNet Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'test_results_summary.png'))
    plt.close()
    
    print("\nAll visualizations completed!")
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main() 