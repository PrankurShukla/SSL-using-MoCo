# %% [markdown]
# # Post-MoCo Training Steps for Kvasir-Instrument Segmentation
# 
# This notebook implements the steps after MoCo pretraining, including:
# 1. Evaluating the pretrained model
# 2. Fine-tuning for segmentation
# 3. Visualizing results

# %%
# Install required packages
!pip install albumentations tensorboard scikit-learn matplotlib

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Set up Kaggle paths
DATA_DIR = "/kaggle/input/kvasir-dataset/kvasir-instrument"  # Modified for Kaggle
RESULTS_DIR = "/kaggle/working/results"       # Modified for Kaggle
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
# Load the pretrained MoCo model
def load_pretrained_moco(checkpoint_path):
    """
    Load the pretrained MoCo model
    
    Args:
        checkpoint_path: Path to the pretrained model checkpoint
    
    Returns:
        model: The loaded MoCo model
    """
    # Create a new model
    model = MoCo(dim=128, K=4096, m=0.999, T=0.07, arch='resnet50')
    
    # Load the pretrained weights
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    return model

# %%
# Function to extract features from images
def extract_features(model, image_path):
    """
    Extract features from an image using the pretrained model
    
    Args:
        model: The pretrained MoCo model
        image_path: Path to the image
    
    Returns:
        features: The extracted features
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    image = transform(image=image)['image'].unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model.encoder_q(image)
    
    return features.cpu().numpy()

# %%
# Function to visualize features using t-SNE
def visualize_features(features, image_paths, title="t-SNE visualization of pretrained features"):
    """
    Visualize features using t-SNE
    
    Args:
        features: List of feature vectors
        image_paths: List of image paths
        title: Title for the plot
    """
    # Print debugging information
    print(f"Number of features: {len(features)}")
    print(f"Number of image paths: {len(image_paths)}")
    
    # Flatten features
    flat_features = np.vstack([f.reshape(1, -1) for f in features])
    print(f"Shape of flattened features: {flat_features.shape}")
    
    # Check if we have enough samples for t-SNE
    n_samples = flat_features.shape[0]
    print(f"Number of samples: {n_samples}")
    
    if n_samples < 2:
        print(f"Not enough samples ({n_samples}) for visualization. Need at least 2 samples.")
        return
    
    # If we have very few samples, use a simpler visualization
    if n_samples < 5:
        print(f"Only {n_samples} samples available. Using simple feature visualization instead of t-SNE.")
        
        # Create a simple visualization of the features
        plt.figure(figsize=(12, 6))
        
        # Plot feature values for each sample
        for i, feature in enumerate(flat_features):
            plt.plot(feature[:50], label=os.path.basename(image_paths[i]))
        
        plt.title(f"Feature values for {n_samples} samples")
        plt.xlabel("Feature dimension")
        plt.ylabel("Feature value")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Also show a heatmap of the features
        plt.figure(figsize=(10, 6))
        plt.imshow(flat_features, aspect='auto', cmap='viridis')
        plt.colorbar(label='Feature value')
        plt.title(f"Feature heatmap for {n_samples} samples")
        plt.xlabel("Feature dimension")
        plt.ylabel("Sample")
        plt.yticks(range(n_samples), [os.path.basename(p) for p in image_paths])
        plt.tight_layout()
        plt.show()
        
        return
    
    # For 5 or more samples, use t-SNE
    # Calculate appropriate perplexity (must be less than n_samples)
    # Ensure perplexity is at least 1 and less than n_samples
    perplexity = max(1, min(30, n_samples - 1))
    print(f"Using perplexity: {perplexity}")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embedded = tsne.fit_transform(flat_features)
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(embedded[:, 0], embedded[:, 1])
    for i, txt in enumerate([os.path.basename(p) for p in image_paths]):
        plt.annotate(txt, (embedded[i, 0], embedded[i, 1]))
    plt.title(title)
    plt.show()

# %%
# Segmentation dataset class
class KvasirInstrumentSegmentationDataset(Dataset):
    """
    Dataset class for Kvasir-Instrument segmentation
    """
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')])
        
        # Split dataset
        if split == 'train':
            self.image_files = self.image_files[:int(0.8 * len(self.image_files))]
        elif split == 'val':
            self.image_files = self.image_files[int(0.8 * len(self.image_files)):]
        
        if transform is None:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = transform
            
        self.mask_transform = A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
        
        # Read image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        if self.mask_transform:
            transformed = self.mask_transform(image=mask)
            mask = transformed['image']
        
        return image, mask

# %%
# Segmentation model class
class SegmentationModel(nn.Module):
    """
    Segmentation model with pretrained encoder
    """
    def __init__(self, encoder, num_classes=1):
        super(SegmentationModel, self).__init__()
        self.encoder = encoder
        
        # Create a segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Get features from the encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        
        # Apply segmentation head
        x = self.segmentation_head(x)
        
        # Upsample to original size
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return x

# %%
# Function to create segmentation model from pretrained MoCo
def create_segmentation_model_from_moco(moco_model, num_classes=1):
    """
    Create a segmentation model from a pretrained MoCo model
    
    Args:
        moco_model: The pretrained MoCo model
        num_classes: Number of classes for segmentation
    
    Returns:
        model: The segmentation model
    """
    # Get the encoder
    encoder = moco_model.encoder_q
    
    # Create segmentation model
    segmentation_model = SegmentationModel(encoder, num_classes)
    
    return segmentation_model

# %%
# Loss functions for segmentation
def dice_loss(pred, target):
    """
    Dice loss for segmentation
    
    Args:
        pred: Predicted segmentation
        target: Target segmentation
    
    Returns:
        loss: Dice loss
    """
    smooth = 1.0
    pred = torch.sigmoid(pred)
    # Ensure target is float
    target = target.float()
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def bce_loss(pred, target):
    """
    Binary cross entropy loss for segmentation
    
    Args:
        pred: Predicted segmentation
        target: Target segmentation
    
    Returns:
        loss: BCE loss
    """
    # Ensure target is float
    target = target.float()
    return nn.BCEWithLogitsLoss()(pred, target)

# %%
# Function to train segmentation model
def train_segmentation_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4):
    """
    Train the segmentation model
    
    Args:
        model: The segmentation model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        model: The trained model
    """
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(RESULTS_DIR, 'segmentation_logs'))
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Segmentation Epoch {epoch+1}/{num_epochs}')
        for images, masks in progress_bar:
            images = images.to(device)
            # Ensure masks are float
            masks = masks.to(device).float()
            
            # Forward pass
            outputs = model(images)
            loss = dice_loss(outputs, masks) + bce_loss(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                # Ensure masks are float
                masks = masks.to(device).float()
                
                # Forward pass
                outputs = model(images)
                loss = dice_loss(outputs, masks) + bce_loss(outputs, masks)
                
                # Update validation loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Print metrics
        print(f"Segmentation Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(RESULTS_DIR, f'segmentation_model_epoch_{epoch+1}.pth')
            )
            print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(RESULTS_DIR, 'best_segmentation_model.pth')
            )
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    writer.close()
    print("Segmentation training completed!")
    
    return model

# %%
# Function to evaluate segmentation model
def evaluate_segmentation_model(model, test_loader, num_samples=5):
    """
    Evaluate the segmentation model
    
    Args:
        model: The segmentation model
        test_loader: Test data loader
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            if i >= num_samples:  # Just show a few examples
                break
                
            image = image.to(device)
            mask = mask.to(device)
            
            output = model(image)
            pred = torch.sigmoid(output) > 0.5
            
            # Convert to numpy for visualization
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            mask_np = mask[0].cpu().numpy().squeeze()
            pred_np = pred[0].cpu().numpy().squeeze()
            
            # Denormalize image
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            # Plot
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask_np, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred_np, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

# %%
# Function to compare with baseline (no pretraining)
def train_baseline_model(train_loader, val_loader, num_epochs=50, lr=1e-4):
    """
    Train a baseline model without pretraining
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        model: The trained baseline model
    """
    # Create a model from scratch
    encoder = models.resnet50(pretrained=False)
    baseline_model = SegmentationModel(encoder)
    baseline_model = baseline_model.to(device)
    
    # Train the model
    return train_segmentation_model(baseline_model, train_loader, val_loader, num_epochs, lr)

# %%
# Main execution
if __name__ == "__main__":
    # Check if the pretrained model exists
    pretrained_model_path = os.path.join(RESULTS_DIR, 'final_model.pth')
    if not os.path.exists(pretrained_model_path):
        print(f"Pretrained model not found at {pretrained_model_path}")
        print("Please run the MoCo pretraining first.")
    else:
        print(f"Loading pretrained model from {pretrained_model_path}")
        moco_model = load_pretrained_moco(pretrained_model_path)
        
        # Step 1: Evaluate the pretrained model
        print("\n=== Step 1: Evaluating the pretrained model ===")
        
        # Extract features from a few images
        image_dir = os.path.join(DATA_DIR, 'images')
        sample_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)[:5]]
        features = [extract_features(moco_model, img_path) for img_path in sample_images]
        
        # Visualize features
        visualize_features(features, sample_images)
        
        # Step 2: Create segmentation datasets
        print("\n=== Step 2: Creating segmentation datasets ===")
        
        train_dataset = KvasirInstrumentSegmentationDataset(DATA_DIR, split='train')
        val_dataset = KvasirInstrumentSegmentationDataset(DATA_DIR, split='val')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Step 3: Create and train segmentation model
        print("\n=== Step 3: Creating and training segmentation model ===")
        
        segmentation_model = create_segmentation_model_from_moco(moco_model)
        segmentation_model = segmentation_model.to(device)
        
        # Step 4: Train the segmentation model
        print("\n=== Training segmentation model ===")
        
        # Training parameters
        num_epochs = 50
        learning_rate = 1e-4
        
        # Train the model
        segmentation_model = train_segmentation_model(
            segmentation_model, 
            train_loader, 
            val_loader, 
            num_epochs=num_epochs, 
            lr=learning_rate
        )
        
        # Step 5: Evaluate the segmentation model
        print("\n=== Evaluating segmentation model ===")
        
        # Load the best model for evaluation
        print("\n=== Loading best model for evaluation ===")
        
        best_model_path = os.path.join(RESULTS_DIR, 'best_segmentation_model.pth')
        segmentation_model.load_state_dict(torch.load(best_model_path, map_location=device))
        segmentation_model.eval()
        
        # Create a test dataset
        print("\n=== Creating test dataset ===")
        
        test_dataset = KvasirInstrumentSegmentationDataset(DATA_DIR, split='val')
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )
        
        # Evaluate
        evaluate_segmentation_model(segmentation_model, test_loader, num_samples=5)
        
        # Step 6: Train baseline model for comparison
        print("\n=== Step 5: Training baseline model for comparison ===")
        
        baseline_model = train_baseline_model(train_loader, val_loader)
        
        # Evaluate baseline model
        print("\n=== Evaluating baseline model ===")
        
        # Load the best baseline model
        baseline_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'best_segmentation_model.pth')))
        baseline_model.eval()
        
        # Evaluate
        evaluate_segmentation_model(baseline_model, test_loader)
        
        # Step 7: Export the Model for Deployment
        print("\n=== Exporting model for deployment ===")
        
        # Save the model in a format suitable for deployment
        torch.save(
            segmentation_model.state_dict(),
            os.path.join(RESULTS_DIR, 'segmentation_model_for_deployment.pth')
        )
        
        # Create a simplified version of the model for inference
        class InferenceModel(nn.Module):
            def __init__(self, model):
                super(InferenceModel, self).__init__()
                self.model = model
                
            def forward(self, x):
                return torch.sigmoid(self.model(x))
        
        inference_model = InferenceModel(segmentation_model)
        inference_model.eval()
        
        # Save the inference model
        torch.save(
            inference_model.state_dict(),
            os.path.join(RESULTS_DIR, 'inference_model.pth')
        )
        
        print("Model exported successfully!")
        
        # Step 8: Create a Simple Inference Script
        print("\n=== Creating inference script ===")
        
        inference_script = """
        import torch
        import torch.nn as nn
        import cv2
        import numpy as np
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Define the model architecture
        class SegmentationModel(nn.Module):
            def __init__(self, encoder, num_classes=1):
                super(SegmentationModel, self).__init__()
                self.encoder = encoder
                
                # Create a segmentation head
                self.segmentation_head = nn.Sequential(
                    nn.Conv2d(2048, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, num_classes, kernel_size=1)
                )
                
            def forward(self, x):
                # Get features from the encoder
                x = self.encoder.conv1(x)
                x = self.encoder.bn1(x)
                x = self.encoder.relu(x)
                x = self.encoder.maxpool(x)
                
                x = self.encoder.layer1(x)
                x = self.encoder.layer2(x)
                x = self.encoder.layer3(x)
                x = self.encoder.layer4(x)
                
                # Apply segmentation head
                x = self.segmentation_head(x)
                
                # Upsample to original size
                x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                
                return x
        
        class InferenceModel(nn.Module):
            def __init__(self, model):
                super(InferenceModel, self).__init__()
                self.model = model
                
            def forward(self, x):
                return torch.sigmoid(self.model(x))
        
        def predict(image_path, model_path, device='cuda'):
            # Load the model
            model = torch.load(model_path, map_location=device)
            model.eval()
            
            # Load and preprocess the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            
            image = transform(image=image)['image'].unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(image)
                prediction = (output > 0.5).float()
            
            # Convert to numpy
            prediction = prediction.cpu().numpy().squeeze()
            
            return prediction
        
        # Example usage
        # prediction = predict('path/to/image.jpg', 'path/to/model.pth')
        # cv2.imwrite('prediction.png', prediction * 255)
        """
        
        # Save the inference script
        with open(os.path.join(RESULTS_DIR, 'inference_script.py'), 'w') as f:
            f.write(inference_script)
        
        print("Inference script created successfully!")
        
        # Step 9: Create a Visualization Script
        print("\n=== Creating visualization script ===")
        
        visualization_script = """
        import torch
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from inference_script import predict
        
        def visualize_prediction(image_path, model_path, output_path=None):
            # Make prediction
            prediction = predict(image_path, model_path)
            
            # Load original image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize prediction to match original image
            prediction = cv2.resize(prediction, (image.shape[1], image.shape[0]))
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(prediction, cmap='gray')
            plt.title('Segmentation Mask')
            plt.axis('off')
            
            # Create overlay
            overlay = image.copy()
            overlay[prediction > 0.5] = [255, 0, 0]  # Red for instrument
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                print(f"Visualization saved to {output_path}")
            else:
                plt.show()
        
        # Example usage
        # visualize_prediction('path/to/image.jpg', 'path/to/model.pth', 'visualization.png')
        """
        
        # Save the visualization script
        with open(os.path.join(RESULTS_DIR, 'visualization_script.py'), 'w') as f:
            f.write(visualization_script)
        
        print("Visualization script created successfully!")
        
        # Step 10: Create a README for Deployment
        print("\n=== Creating README for deployment ===")
        
        readme_content = """
        # Kvasir-Instrument Segmentation Model
        
        This repository contains a pretrained model for surgical instrument segmentation in endoscopic images from the Kvasir-Instrument dataset.
        
        ## Model Details
        
        - **Architecture**: ResNet-50 encoder with custom segmentation head
        - **Pretraining**: Momentum Contrast (MoCo) pretraining
        - **Task**: Binary segmentation of surgical instruments
        
        ## Files
        
        - `segmentation_model_for_deployment.pth`: The full model for fine-tuning
        - `inference_model.pth`: A simplified model for inference
        - `inference_script.py`: Script for making predictions
        - `visualization_script.py`: Script for visualizing predictions
        
        ## Usage
        
        ### Making Predictions
        
        ```python
        from inference_script import predict
        
        # Make a prediction
        prediction = predict('path/to/image.jpg', 'path/to/model.pth')
        
        # Save the prediction
        import cv2
        cv2.imwrite('prediction.png', prediction * 255)
        ```
        
        ### Visualizing Predictions
        
        ```python
        from visualization_script import visualize_prediction
        
        # Visualize a prediction
        visualize_prediction('path/to/image.jpg', 'path/to/model.pth', 'visualization.png')
        ```
        
        ## Dataset
        
        The model was trained on the Kvasir-Instrument dataset, which contains endoscopic images with surgical instrument annotations.
        
        ## References
        
        - [MoCo v2: Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)
        - [Kvasir-Instrument Dataset](https://datasets-server.huggingface.co/datasets/kvasir-instrument)
        """
        
        # Save the README
        with open(os.path.join(RESULTS_DIR, 'README.md'), 'w') as f:
            f.write(readme_content)
        
        print("README created successfully!")
        
        print("\n=== All steps completed! ===")
        print("Your segmentation model is ready for deployment.")
        print(f"All files have been saved to {RESULTS_DIR}") 