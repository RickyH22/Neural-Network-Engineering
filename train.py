import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from layers import LearnedAffine

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

class FashionMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.learned_affine = LearnedAffine(128)  # Custom layer
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.learned_affine(x)
        x = self.fc2(x)
        return x


def train_epoch(model, device, train_loader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} with seed {SEED}\n")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_loader = DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )
    
    test_loader = DataLoader(
        datasets.FashionMNIST('./data', train=False, download=True, transform=transform),
        batch_size=64, shuffle=False
    )
    
    # Model, loss, optimizer, scheduler
    model = FashionMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=3, steps_per_epoch=len(train_loader)
    )
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Training loop
    best_acc = 0.0
    for epoch in range(1, 4):
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer, scheduler)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        
        print(f"Epoch {epoch}/3 - Train: {train_loss:.3f} loss, {train_acc:.1f}% | "
              f"Test: {test_loss:.3f} loss, {test_acc:.1f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"\nBest accuracy: {best_acc:.2f}%")
    print("Metrics saved in metrics.md")


if __name__ == "__main__":
    main()
