# %% [markdown]
# # Improved MoCo Pretraining and Segmentation for Kvasir-Instrument Dataset
#
# This notebook implements an improved version of MoCo pretraining followed by surgical instrument segmentation.
# Phase 1: MoCo Pretraining
# Phase 2: Segmentation Fine-tuning

# %% [code]
# Install required packages
!pip install -q albumentations
!pip install -q tensorboard
!pip install -q segmentation-models-pytorch
!pip install -q monai

# %% [code]
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
warnings.filterwarnings('ignore')

# %% [code]
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

# %% [markdown]
# ## 1. MoCo Implementation

# %% [code]
class MoCo(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = models.resnet50(weights=None)
        self.encoder_k = models.resnet50(weights=None)

        # remove the final fc layer
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

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
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
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            self.queue[:, ptr:] = keys[:, :self.K-ptr].T
            self.queue[:, :batch_size-(self.K-ptr)] = keys[:, self.K-ptr:].T
            ptr = batch_size-(self.K-ptr)
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
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
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

# %% [code]
class MoCoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load all image paths
        self.images = sorted([
            f for f in os.listdir(os.path.join(data_dir, 'images')) 
            if f.endswith(('.jpg', '.png'))
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, 'images', img_name)
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            aug1 = self.transform(image=image)['image']
            aug2 = self.transform(image=image)['image']
            return aug1, aug2
        return image, image

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

# %% [markdown]
# ## 2. MoCo Training

# %% [code]
def train_moco(data_dir, num_epochs=200):
    # Create dataset and dataloader
    dataset = MoCoDataset(data_dir, transform=get_moco_augmentation())
    dataloader = DataLoader(
        dataset, 
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    # Create MoCo model
    model = MoCo().to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.encoder_q.parameters(), lr=1e-3)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(RESULTS_DIR, 'moco_logs'))
    
    print(f'Dataset size: {len(dataset)}')
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for im_q, im_k in pbar:
                im_q, im_k = im_q.to(device), im_k.to(device)
                
                # Forward pass
                output, target = model(im_q, im_k)
                loss = criterion(output, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(RESULTS_DIR, f'moco_checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(RESULTS_DIR, 'final_moco_model.pth'))
    
    writer.close()
    print('MoCo pretraining completed!')
    return model

# %% [markdown]
# ## 1. Improved Dataset and Augmentations

# %% [code]
class ImprovedKvasirDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load all image paths
        self.images = sorted([f for f in os.listdir(os.path.join(data_dir, 'images')) 
                            if f.endswith(('.jpg', '.png'))])
        
        # Split dataset
        if split == 'train':
            self.images = self.images[:int(0.8 * len(self.images))]
        elif split == 'val':
            self.images = self.images[int(0.8 * len(self.images)):int(0.9 * len(self.images))]
        else:  # test
            self.images = self.images[int(0.9 * len(self.images)):]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, 'images', img_name)
        mask_path = os.path.join(self.data_dir, 'masks', img_name)
        
        # Read image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize mask to binary
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

# Advanced augmentations
def get_training_augmentation():
    return A.Compose([
        A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
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
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# %% [markdown]
# ## 2. Improved Model Architecture

# %% [code]
class ImprovedSegmentationModel(nn.Module):
    def __init__(self, moco_path=None):
        super().__init__()
        # Load MoCo pretrained model if available
        if moco_path and os.path.exists(moco_path):
            print("Loading MoCo pretrained weights...")
            moco_state = torch.load(moco_path)
            moco_model = MoCo()
            moco_model.load_state_dict(moco_state['model_state_dict'])
            encoder = moco_model.encoder_q
        else:
            print("Using ImageNet pretrained weights...")
            encoder = models.resnet50(weights='IMAGENET1K_V2')
        
        # Remove the final fc layer
        encoder = nn.Sequential(*list(encoder.children())[:-2])
        
        self.model = smp.UnetPlusPlus(
            encoder_name='resnet50',
            encoder_weights=None,  # We'll load our pretrained weights
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # Load pretrained encoder weights
        if moco_path and os.path.exists(moco_path):
            self.model.encoder = encoder
        
    def forward(self, x):
        return self.model(x)

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Dice loss
        dice = 1 - (2 * (pred * target).sum() + self.smooth) / \
               (pred.sum() + target.sum() + self.smooth)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        
        # Combined loss
        return dice + bce

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Extract boundaries
        pred_boundary = F.conv2d(pred, self.laplacian_kernel, padding=1)
        target_boundary = F.conv2d(target, self.laplacian_kernel, padding=1)
        
        # Calculate boundary loss
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        return boundary_loss

# %% [markdown]
# ## 3. Training Functions with Improvements

# %% [code]
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    epoch_loss = 0
    
    with tqdm(train_loader, desc=f'Training Epoch {epoch + 1}') as pbar:
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (pbar.n + 1)})
    
    if scheduler is not None:
        scheduler.step(epoch_loss / len(train_loader))
    
    return epoch_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # Calculate Dice score
            pred = torch.sigmoid(outputs) > 0.5
            dice = (2 * (pred * masks).sum()) / (pred.sum() + masks.sum() + 1e-8)
            dice_scores.append(dice.item())
    
    return val_loss / len(val_loader), np.mean(dice_scores)

def post_process_prediction(pred):
    # Convert to numpy
    pred = pred.cpu().numpy().squeeze()
    
    # Apply threshold
    binary = (pred > 0.5).astype(np.uint8)
    
    # Remove small objects
    binary = ndimage.binary_opening(binary, structure=np.ones((3,3)))
    
    # Remove small holes
    binary = ndimage.binary_closing(binary, structure=np.ones((3,3)))
    
    # Keep only the largest connected component
    labels, num_features = ndimage.label(binary)
    if num_features > 0:
        sizes = ndimage.sum(binary, labels, range(1, num_features + 1))
        max_label = sizes.argmax() + 1
        binary = (labels == max_label).astype(np.uint8)
    
    return torch.from_numpy(binary).float()

# %% [markdown]
# ## 4. Main Training Loop

# %% [code]
def main():
    # Create datasets
    train_dataset = ImprovedKvasirDataset(
        DATA_DIR,
        split='train',
        transform=get_training_augmentation()
    )
    
    val_dataset = ImprovedKvasirDataset(
        DATA_DIR,
        split='val',
        transform=get_validation_augmentation()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = ImprovedSegmentationModel().to(device)
    
    # Initialize loss functions
    dice_bce_loss = DiceBCELoss()
    boundary_loss = BoundaryLoss()
    
    def combined_loss(pred, target):
        return dice_bce_loss(pred, target) + 0.5 * boundary_loss(pred, target)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(RESULTS_DIR, 'logs'))
    
    # Training loop
    num_epochs = 50
    best_dice = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, combined_loss,
            optimizer, scheduler, epoch
        )
        
        # Validate
        val_loss, dice_score = validate(model, val_loader, combined_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', dice_score, epoch)
        
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Dice Score: {dice_score:.4f}')
        
        # Save best model
        if dice_score > best_dice:
            best_dice = dice_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(RESULTS_DIR, 'best_model.pth'))
            print(f'Saved best model with Dice score: {best_dice:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(RESULTS_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print('Training completed!')
    
    return model

# %% [markdown]
# ## 5. Visualization and Evaluation

# %% [code]
def visualize_predictions(model, dataset, num_samples=5):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            image = image.to(device)
            output = model(image)
            pred = torch.sigmoid(output)
            pred = post_process_prediction(pred)
            
            # Denormalize image
            image = image.cpu().squeeze().permute(1,2,0)
            image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            image = image.numpy()
            image = np.clip(image, 0, 1)
            
            # Plot
            axes[i, 0].imshow(image)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.squeeze(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred.squeeze(), cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Run Training

# %% [code]
if __name__ == "__main__":
    # Phase 1: MoCo Pretraining
    print("=== Phase 1: MoCo Pretraining ===")
    moco_model = train_moco(DATA_DIR)
    
    # Phase 2: Segmentation Training
    print("\n=== Phase 2: Segmentation Training ===")
    model = main()
    
    # Create test dataset
    test_dataset = ImprovedKvasirDataset(
        DATA_DIR,
        split='test',
        transform=get_validation_augmentation()
    )
    
    # Visualize results
    visualize_predictions(model, test_dataset) 