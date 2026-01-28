import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, models, transforms
import os
import argparse
import time
from tqdm import tqdm
from PIL import Image
import sys
import numpy as np
from src.detection.core import find_support_boundary

# Common device function
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# --- Training Logic ---

class AugmentedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        
        # --- Support Detection & Smart Cropping ---
        # Convert to numpy for support detection
        x_np = np.array(x)
        try:
            # Detect support boundary (returns Y-coordinate of bottom of support)
            support_y = find_support_boundary(x_np)
        except Exception:
            support_y = 0

        w, h = x.size
        # Ensure we have enough height for 224 crop
        if h - support_y < 224:
            start_y = max(0, h - 224)
        else:
            start_y = support_y
            
        # Crop to the region of interest (below support)
        x = x.crop((0, start_y, w, h))
        
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def train_noise_classifier(data_dir, output_model, epochs=10, batch_size=8, learning_rate=0.001):
    device = get_device()
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    # Data Augmentation for Training
    train_transforms = transforms.Compose([
        transforms.RandomCrop(224), # Random crop at full resolution to see noise
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Standard Normalization for Validation
    val_transforms = transforms.Compose([
        transforms.RandomCrop(224), # Random crop for validation too effectively
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        return

    # Load Dataset (without transforms initially)
    try:
        full_dataset = datasets.ImageFolder(data_dir) 
    except FileNotFoundError:
        print(f"Error: Could not load data from {data_dir}. Ensure 'clean' and 'noisy' folders exist.")
        return

    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    
    # 80/20 Train/Val Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # Wrap subsets with specific transforms
    train_dataset = AugmentedDataset(train_subset, train_transforms)
    val_dataset = AugmentedDataset(val_subset, val_transforms)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    effective_batch_size = max(batch_size, 64) # Force higher for RTX 4080
    print(f"Training with batch size: {effective_batch_size}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=effective_batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=effective_batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize Model (ResNet18)
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    # --- TRANSFER LEARNING: Freeze backbone EXCEPT layer 4 ---
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Use Adam for faster convergence in fine-tuning
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate, weight_decay=1e-4)

    # AMP Setup
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    # Training Loop
    since = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc=phase, leave=False):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if scaler:
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), output_model)

    print(f'Best val Acc: {best_acc:4f}')

# --- Detection/Inference Logic ---

class SimpleImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        self.image_paths = sorted([
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if os.path.splitext(f)[1].lower() in self.image_extensions
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # --- Support Detection & Smart Cropping (Inference) ---
        img_np = np.array(image)
        try:
             support_y = find_support_boundary(img_np)
        except:
             support_y = 0
             
        w, h = image.size
        if h - support_y < 224:
             start_y = max(0, h - 224)
        else:
             start_y = support_y
        
        image = image.crop((0, start_y, w, h))

        if self.transform:
            image = self.transform(image)
        return image, img_path

def detect_noise_in_folder(folder_path, model_path):
    device = get_device()
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        # Consistently use original resolution (RandomCrop in model means we should NOT resize here)
        # But for full image inference, we can just use RandomCrop(224) to get a sample
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleImageDataset(folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)
    
    noisy_count = 0
    total_count = 0
    classes = ['Clean', 'NOISY'] 
    
    with torch.no_grad():
        for inputs, paths in tqdm(loader, unit="batch"):
            inputs = inputs.to(device)
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            
            for i in range(len(paths)):
                pred_idx = preds[i].item()
                label = classes[pred_idx]
                if label == 'NOISY': noisy_count += 1
                total_count += 1

    print(f"Result: {noisy_count}/{total_count} images flagged as NOISY.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise Detection Utilities")
    subparsers = parser.add_subparsers(dest="command")
    
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data_dir", type=str, required=True)
    train_parser.add_argument("--output", type=str, default="noise_classifier.pth")
    train_parser.add_argument("--epochs", type=int, default=10)
    
    detect_parser = subparsers.add_parser("detect", help="Detect noise in folder")
    detect_parser.add_argument("folder", type=str)
    detect_parser.add_argument("--model", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_noise_classifier(args.data_dir, args.output, epochs=args.epochs)
    elif args.command == "detect":
        detect_noise_in_folder(args.folder, args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    # Simple CLI dispatch
    parser = argparse.ArgumentParser(description="Noise Detection Utilities")
    subparsers = parser.add_subparsers(dest="command")
    
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data_dir", type=str, required=True)
    train_parser.add_argument("--output", type=str, default="noise_classifier.pth")
    train_parser.add_argument("--epochs", type=int, default=10)
    
    detect_parser = subparsers.add_parser("detect", help="Detect noise in folder")
    detect_parser.add_argument("folder", type=str)
    detect_parser.add_argument("--model", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_noise_classifier(args.data_dir, args.output, epochs=args.epochs)
    elif args.command == "detect":
        detect_noise_in_folder(args.folder, args.model)
    else:
        parser.print_help()
