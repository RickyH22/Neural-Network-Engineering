"""
Training script for FashionMNIST classification with custom LearnedAffine layer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from layers import LearnedAffine


# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class FashionMNISTNet(nn.Module):
    """
    Simple CNN with custom LearnedAffine layer for FashionMNIST classification.
    """
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.learned_affine = LearnedAffine(128)  # Custom layer
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC block with custom LearnedAffine
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.learned_affine(x)  # Apply custom affine transformation
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


def train_epoch(model, device, train_loader, criterion, optimizer, epoch, scheduler=None):
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        device: Device to train on (CPU/CUDA)
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        epoch: Current epoch number
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        Average training loss for the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Step scheduler if using OneCycleLR (per-batch scheduling)
        if scheduler is not None:
            scheduler.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100. * correct / total:.2f}%')
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network model
        device: Device to evaluate on (CPU/CUDA)
        test_loader: DataLoader for test data
        criterion: Loss function
    
    Returns:
        Test loss and accuracy
    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    """Main training function."""
    print("=" * 70)
    print("FashionMNIST Training with Custom LearnedAffine Layer")
    print("=" * 70)
    
    # Hyperparameters
    batch_size = 64
    epochs = 3
    learning_rate = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Seed: {SEED}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load FashionMNIST dataset
    print("\nLoading FashionMNIST dataset...")
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = FashionMNISTNet().to(device)
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: AdamW
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler: OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    
    print(f"\nOptimizer: AdamW")
    print(f"Scheduler: OneCycleLR (max_lr={learning_rate * 10})")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_acc = 0.0
    training_history = []
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, criterion, optimizer, epoch, scheduler
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save training history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'time': epoch_time
        })
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ New best model saved! (Acc: {best_acc:.2f}%)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    
    # Print training history
    print("\nTraining History:")
    print("-" * 70)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12}")
    print("-" * 70)
    for h in training_history:
        print(f"{h['epoch']:<8} {h['train_loss']:<12.4f} {h['train_acc']:<12.2f} "
              f"{h['test_loss']:<12.4f} {h['test_acc']:<12.2f}")
    print("-" * 70)
    
    return model, training_history


if __name__ == "__main__":
    main()
