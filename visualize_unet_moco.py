import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# Paths
DATA_DIR = '/kaggle/input/kvasir-dataset/kvasir-instrument'
MODEL_PATH = '/kaggle/working/results/best_model_unet_moco_20.pth'  # Path to your UNet MoCo model
SAVE_DIR = '/kaggle/working/results/unet_visualizations'

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder (contracting path)
        self.enc1 = self.double_conv(n_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)
        
        # Decoder (expansive path)
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(1024, 512)
        
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(512, 256)
        
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(256, 128)
        
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool(e1))
        e3 = self.enc3(self.maxpool(e2))
        e4 = self.enc4(self.maxpool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.maxpool(e4))
        
        # Decoder
        d4 = self.up_conv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up_conv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up_conv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up_conv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class KvasirDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.image_files = [f for f in self.image_files if os.path.exists(
            os.path.join(self.mask_dir, f.replace('.jpg', '.png')))]
        
        print(f"Found {len(self.image_files)} valid image-mask pairs")
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace('.jpg', '.png'))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask, self.image_files[idx]
    
    def __len__(self):
        return len(self.image_files)

def get_transform():
    return A.Compose([
        A.Resize(256, 256),  # UNet typically uses power of 2 sizes
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def visualize_sample(image, mask, prediction, filename, save_path=None):
    """Simple visualization showing original images and raw masks"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(132)
    plt.imshow(mask.cpu().numpy(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Model prediction
    plt.subplot(133)
    plt.imshow(prediction.cpu().numpy(), cmap='gray')
    plt.title('Prediction (20% Training)')
    plt.axis('off')
    
    plt.suptitle(filename)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNet(n_channels=3, n_classes=1)
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Clean state dict keys
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Create dataset and get samples
    dataset = KvasirDataset(DATA_DIR, transform=get_transform())
    indices = range(min(5, len(dataset)))  # Show first 5 samples
    
    print("\nGenerating visualizations...")
    with torch.no_grad():
        for idx in indices:
            image, mask, filename = dataset[idx]
            
            # Get prediction
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            pred = torch.sigmoid(output).squeeze()
            
            # Save visualization
            save_path = os.path.join(SAVE_DIR, f'unet_pred_{filename[:-4]}.png')
            visualize_sample(image, mask, pred, filename, save_path)
            print(f"Saved visualization for {filename}")

if __name__ == '__main__':
    main() 