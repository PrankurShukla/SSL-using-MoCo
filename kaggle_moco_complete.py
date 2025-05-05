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
import matplotlib.pyplot as plt

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
# MoCo model implementation
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, dim=128, K=4096, m=0.999, T=0.07, arch='resnet50'):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 4096)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = self._build_encoder(arch, dim)
        self.encoder_k = self._build_encoder(arch, dim)

        # initialize the key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _build_encoder(self, arch, dim):
        """
        Build the encoder network
        """
        if arch == 'resnet50':
            encoder = models.resnet50(pretrained=True)
            # modify the final layer
            encoder.fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, dim)
            )
        else:
            raise NotImplementedError(f"Architecture {arch} not implemented")
        
        return encoder

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
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
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

# %%
# Dataset class for MoCo pretraining
class KvasirInstrumentMoCoDataset(Dataset):
    """
    Dataset class for MoCo pretraining on Kvasir-Instrument
    Returns two augmented views of the same image
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')])
        
        self.transform = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.2, 1.0)),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.GaussianBlur(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create two augmented views
        view1 = self.transform(image=image)['image']
        view2 = self.transform(image=image)['image']
        
        return view1, view2

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
    for batch in progress_bar:
        # Get the two augmented views
        im_q = batch[0].to(device)
        im_k = batch[1].to(device)
        
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

# %%
# Function to load the pretrained model for segmentation
def load_pretrained_model_for_segmentation(checkpoint_path, num_classes=1):
    """
    Load the pretrained MoCo model and add a segmentation head
    
    Args:
        checkpoint_path: Path to the pretrained model checkpoint
        num_classes: Number of classes for segmentation (default: 1 for binary segmentation)
    
    Returns:
        model: The model with segmentation head
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Create a new model with segmentation head
    model = MoCo(dim=128, K=4096, m=0.999, T=0.07, arch='resnet50')
    
    # Load the pretrained weights
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Get the encoder
    encoder = model.encoder_q
    
    # Create a segmentation head
    segmentation_head = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, num_classes, kernel_size=1)
    )
    
    # Create a new model with the encoder and segmentation head
    class SegmentationModel(nn.Module):
        def __init__(self, encoder, segmentation_head):
            super(SegmentationModel, self).__init__()
            self.encoder = encoder
            self.segmentation_head = segmentation_head
            
        def forward(self, x):
            # Get features from the encoder
            features = []
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
    
    # Create the segmentation model
    segmentation_model = SegmentationModel(encoder, segmentation_head)
    
    return segmentation_model

# Example of how to use the pretrained model for segmentation
# segmentation_model = load_pretrained_model_for_segmentation(os.path.join(RESULTS_DIR, 'final_model.pth'))
# segmentation_model = segmentation_model.to(device)
# segmentation_model.eval() 