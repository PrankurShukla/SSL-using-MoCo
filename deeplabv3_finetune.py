import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set paths
DATA_DIR = '/kaggle/input/kvasir-dataset/kvasir-instrument'
RESULTS_DIR = '/kaggle/working/results'
MOCO_CHECKPOINT = os.path.join(RESULTS_DIR, 'final_moco_model.pth')

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

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
            if f.endswith('.jpg')  # Only look for .jpg images
        ])
        
        # Verify corresponding masks exist (with .png extension)
        valid_pairs = []
        for img_name in self.images:
            # Convert jpg to png for mask filename
            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                valid_pairs.append((img_name, mask_name))
            else:
                print(f"Warning: Mask not found for {img_name}, skipping...")
        
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
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
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

class DeepLabV3MoCo(nn.Module):
    def __init__(self, moco_path):
        super().__init__()
        
        # Load MoCo state dict
        moco_state = torch.load(moco_path)
        
        # Create DeepLabV3 model
        self.model = deeplabv3_resnet50(num_classes=1)
        
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
        return output['out']

def dice_loss(pred, target):
    smooth = 1.0
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

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate Dice score
            pred = (torch.sigmoid(outputs) > 0.5).float()
            dice = (2.0 * (pred * masks).sum()) / (pred.sum() + masks.sum() + 1e-8)
            dice_scores.append(dice.item())
            
            total_loss += loss.item()
    
    return total_loss / len(loader), np.mean(dice_scores)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = KvasirSegmentationDataset(
        DATA_DIR,
        split='train',
        transform=get_training_augmentation()
    )
    val_dataset = KvasirSegmentationDataset(
        DATA_DIR,
        split='val',
        transform=get_validation_augmentation()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = DeepLabV3MoCo(MOCO_CHECKPOINT).to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW([
        {'params': model.model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.model.classifier.parameters(), 'lr': 1e-4},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create criterion
    criterion = dice_loss
    
    # Initialize tensorboard
    writer = SummaryWriter(os.path.join(RESULTS_DIR, 'deeplabv3_logs'))
    
    # Training loop
    num_epochs = 50
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, dice_score = validate(model, val_loader, criterion, device)
        
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
            }, os.path.join(RESULTS_DIR, 'best_deeplabv3_model.pth'))
            print(f"Saved best model with Dice score: {best_dice:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(RESULTS_DIR, f'deeplabv3_checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main() 