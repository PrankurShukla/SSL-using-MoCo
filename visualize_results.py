import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Paths
DATA_DIR = '/kaggle/input/kvasir-dataset/kvasir-instrument'
RESULTS_DIR = '/kaggle/working/results'
MOCO_MODEL_PATH = os.path.join(RESULTS_DIR, 'best_model_moco.pth')
BASELINE_MODEL_PATH = os.path.join(RESULTS_DIR, 'best_model_baseline.pth')

class KvasirSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get image and mask paths
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.image_files = [f for f in self.image_files if os.path.exists(
            os.path.join(self.mask_dir, f.replace('.jpg', '.png')))]
        
    def __len__(self):
        return len(self.image_files)
    
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

def get_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def load_model(model_path, device):
    model = deeplabv3_resnet50(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def visualize_predictions(models, dataset, device, num_samples=6, save_path=None):
    """
    Visualize predictions from multiple models side by side
    models: dict of name: model pairs
    """
    # Create figure
    num_cols = len(models) + 2  # Original + GT + each model
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4*num_cols, 4*num_samples))
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask, filename = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            # Original image (denormalized)
            img_display = image.cpu().numpy().transpose(1, 2, 0)
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)
            
            # Plot original image
            axes[row, 0].imshow(img_display)
            axes[row, 0].set_title('Original')
            axes[row, 0].axis('off')
            
            # Plot ground truth
            axes[row, 1].imshow(mask.numpy(), cmap='gray')
            axes[row, 1].set_title('Ground Truth')
            axes[row, 1].axis('off')
            
            # Get and plot predictions from each model
            for col, (name, model) in enumerate(models.items(), start=2):
                output = model(image_tensor)['out']
                pred = torch.sigmoid(output).cpu()
                pred_mask = (pred > 0.5).float().squeeze().numpy()
                
                axes[row, col].imshow(pred_mask, cmap='gray')
                axes[row, col].set_title(f'{name}')
                axes[row, col].axis('off')
            
            # Add filename as row title
            axes[row, 0].set_ylabel(filename, rotation=0, labelpad=50, va='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")
    plt.show()

def calculate_metrics(models, dataset, device):
    """Calculate Dice scores for each model"""
    results = {}
    
    for name, model in models.items():
        dice_scores = []
        
        with torch.no_grad():
            for image, mask, _ in dataset:
                image = image.unsqueeze(0).to(device)
                mask = mask.to(device)
                
                output = model(image)['out']
                pred = (torch.sigmoid(output) > 0.5).float()
                
                dice = (2.0 * (pred * mask).sum()) / (pred.sum() + mask.sum() + 1e-8)
                dice_scores.append(dice.item())
        
        avg_dice = np.mean(dice_scores)
        results[name] = avg_dice
    
    return results

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = KvasirSegmentationDataset(DATA_DIR, transform=get_transform())
    print(f"Dataset size: {len(dataset)}")
    
    # Load models
    models = {
        'MoCo-DeepLabV3': load_model(MOCO_MODEL_PATH, device),
        'Baseline-DeepLabV3': load_model(BASELINE_MODEL_PATH, device)
    }
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(models, dataset, device)
    for name, dice_score in metrics.items():
        print(f"{name} - Average Dice Score: {dice_score:.4f}")
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    save_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
    visualize_predictions(models, dataset, device, num_samples=6, save_path=save_path)

if __name__ == '__main__':
    main() 