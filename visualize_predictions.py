import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# Paths
DATA_DIR = '/kaggle/input/kvasir-dataset/kvasir-instrument'
MODEL_PATH = '/kaggle/working/results/best_model_moco.pth'
SAVE_DIR = '/kaggle/working/results/visualizations'

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
        # Load image and mask
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
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def visualize_sample(image, mask, prediction, filename, save_path=None):
    """Simple visualization showing original images and raw masks"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    # Denormalize image
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth mask - show as is
    plt.subplot(132)
    plt.imshow(mask.cpu().numpy(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Model prediction - show raw prediction
    plt.subplot(133)
    plt.imshow(prediction.cpu().numpy(), cmap='gray')
    plt.title('Prediction')
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
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
    
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
    indices = range(min(5, len(dataset)))  # Just show first 5 samples
    
    print("\nGenerating visualizations...")
    with torch.no_grad():
        for idx in indices:
            image, mask, filename = dataset[idx]
            
            # Get prediction
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)['out']
            pred = torch.sigmoid(output).squeeze()
            
            # Save visualization
            save_path = os.path.join(SAVE_DIR, f'pred_{filename[:-4]}.png')
            visualize_sample(image, mask, pred, filename, save_path)
            print(f"Saved visualization for {filename}")

if __name__ == '__main__':
    main() 