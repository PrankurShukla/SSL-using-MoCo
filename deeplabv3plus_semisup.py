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
    def __init__(self, data_dir, split='train', transform=None):
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
        
        # Split dataset
        if split == 'train':
            valid_pairs = valid_pairs[:int(0.8 * len(valid_pairs))]
        elif split == 'val':
            valid_pairs = valid_pairs[int(0.8 * len(valid_pairs)):]
        
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
        
        return image, mask

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
        print("\nInitial backbone weights (first layer):")
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

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    with tqdm(loader) as pbar:
        for images, masks in pbar:
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

def visualize_predictions(model, dataset, device, num_samples=4, save_dir=None, epoch=None):
    """Visualize predictions for a few samples"""
    model.eval()
    
    # Create a figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    # Get random samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            # Get sample
            image, mask = dataset[sample_idx]
            
            # Add batch dimension
            image = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image)
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
        plt.savefig(os.path.join(save_dir, f'predictions{epoch_str}.png'))
        plt.close()
    else:
        plt.show()

def validate(model, loader, criterion, device, writer=None, epoch=None, save_dir=None):
    model.eval()
    total_loss = 0
    dice_scores = []
    
    # Store some examples for visualization
    vis_images = []
    vis_masks = []
    vis_preds = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate Dice score
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
        writer.add_image('Images/Input', image_grid, epoch)
        writer.add_image('Images/GroundTruth', mask_grid, epoch)
        writer.add_image('Images/Prediction', pred_grid, epoch)
    
    # Save detailed visualizations
    if save_dir is not None:
        os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
        visualize_predictions(
            model, 
            loader.dataset, 
            device, 
            num_samples=4,
            save_dir=os.path.join(save_dir, 'visualizations'),
            epoch=epoch
        )
    
    return total_loss / len(loader), np.mean(dice_scores)

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

def test_model(model, test_loader, device, save_dir=None):
    """Evaluate model on test set and calculate metrics"""
    model.eval()
    total_iou = 0
    total_dice = 0
    all_ious = []
    all_dices = []
    
    # Create directory for saving test results
    if save_dir:
        os.makedirs(os.path.join(save_dir, 'test_results'), exist_ok=True)
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
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
                    
                    plt.suptitle(f'Test Sample {i*len(images) + j}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'test_results', f'test_sample_{i*len(images) + j}.png'))
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
    print(f"\nTest Results:")
    print(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"IoU Range: {min_iou:.4f} - {max_iou:.4f}")
    print(f"Dice Range: {min_dice:.4f} - {max_dice:.4f}")
    
    # Save metrics to file
    if save_dir:
        with open(os.path.join(save_dir, 'test_metrics.txt'), 'w') as f:
            f.write(f"Test Results:\n")
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
        plt.title('Distribution of IoU and Dice Scores on Test Set')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'test_results', 'score_distribution.png'))
        plt.close()
    
    return avg_iou, avg_dice, all_ious, all_dices

def create_confusion_matrix(model, test_loader, device, save_dir=None):
    """Create confusion matrix visualization for the test set"""
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\nCreating confusion matrix...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
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
    plt.title('Confusion Matrix')
    
    # Add metrics text
    plt.text(0.5, -0.3, f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}',
             horizontalalignment='center', transform=plt.gca().transAxes)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'test_results', 'confusion_matrix.png'), bbox_inches='tight')
        plt.close()
        
        # Save metrics to file
        with open(os.path.join(save_dir, 'test_metrics.txt'), 'a') as f:
            f.write(f"\nConfusion Matrix Metrics:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
    else:
        plt.show()
    
    return cm, accuracy, precision, recall, f1

def main():
    # Set random seed
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    full_dataset = KvasirSegmentationDataset(DATA_DIR, transform=get_training_augmentation())
    
    # Split into train (50%), val (20%), and test (30%)
    total_size = len(full_dataset)
    train_size = int(0.5 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Create model
    model = DeepLabV3PlusMoCo(MOCO_CHECKPOINT).to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW([
        {'params': model.model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.model.classifier.parameters(), 'lr': 1e-4},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create criterion
    criterion = dice_loss
    
    # Initialize tensorboard
    writer = SummaryWriter(os.path.join(RESULTS_DIR, 'deeplabv3plus_semisup_logs'))
    
    # Create results and visualization directories
    os.makedirs(os.path.join(RESULTS_DIR, 'visualizations'), exist_ok=True)
    
    # Training loop
    num_epochs = 50
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate with visualizations
        val_loss, dice_score = validate(
            model, 
            val_loader, 
            criterion, 
            device,
            writer=writer,
            epoch=epoch,
            save_dir=RESULTS_DIR
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', dice_score, epoch)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Dice Score: {dice_score:.4f}")
        
        # Save best model
        if dice_score > best_dice:
            best_dice = dice_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(RESULTS_DIR, 'best_deeplabv3plus_semisup_model.pth'))
            print(f"Saved best model with Dice score: {best_dice:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(RESULTS_DIR, f'deeplabv3plus_semisup_checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print("Training completed!")

    # After training, evaluate on test set
    print("\nEvaluating final model on test set...")
    test_iou, test_dice, all_ious, all_dices = test_model(
        model, 
        test_loader, 
        device,
        save_dir=RESULTS_DIR
    )
    
    # Create confusion matrix
    create_confusion_matrix(model, test_loader, device, save_dir=RESULTS_DIR)
    
    # Save final test metrics with methodology summary
    with open(os.path.join(RESULTS_DIR, 'final_test_metrics.txt'), 'w') as f:
        f.write(f"Methodology Summary:\n")
        f.write(f"1. Self-supervised pretraining with MoCo on unlabeled data\n")
        f.write(f"2. Fine-tuning DeepLabV3+ with MoCo pretrained features on 50% labeled data\n")
        f.write(f"3. Validation on 20% of data\n")
        f.write(f"4. Testing on 30% of data\n\n")
        
        f.write(f"Final Test Results:\n")
        f.write(f"Average IoU: {test_iou:.4f}\n")
        f.write(f"Average Dice: {test_dice:.4f}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n")
        
        # Add training history summary
        f.write(f"\nTraining History:\n")
        f.write(f"Best validation Dice score: {best_dice:.4f}\n")
        f.write(f"Final training loss: {train_loss:.4f}\n")
        f.write(f"Final validation loss: {val_loss:.4f}\n")
    
    print("Testing completed!")
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main() 