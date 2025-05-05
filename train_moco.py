import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from moco import MoCo
from dataset import KvasirInstrumentMoCoDataset

def main():
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
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and dataloader
    dataset = KvasirInstrumentMoCoDataset(root_dir='path_to_kvasir_dataset')
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
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, _ in progress_bar:
            # Get the two augmented views
            im_q = images[0].to(device)
            im_k = images[1].to(device)
            
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
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(
                checkpoint,
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    writer.close()

if __name__ == '__main__':
    main() 