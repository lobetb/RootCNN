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

# Common device function
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# --- Training Logic ---

def train_noise_classifier(data_dir, output_model, epochs=10, batch_size=8, learning_rate=0.001):
    device = get_device()
    print(f"Using device: {device}")

    # Data Augmentation and Normalization
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        print(f"Please create '{data_dir}' with 'clean' and 'noisy' subfolders.")
        return

    # Load Dataset
    try:
        full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    except FileNotFoundError:
        print(f"Error: Could not load data from {data_dir}. Ensure 'clean' and 'noisy' folders exist.")
        return

    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    
    # 80/20 Train/Val Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize Model (ResNet18)
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        # Fallback for older torchvision versions
        model = models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Binary classification
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training Loop
    since = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), output_model)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    print(f'Model saved to {output_model}')


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
        image = Image.open(img_path).convert('RGB') # ResNet expects 3 channels
        if self.transform:
            image = self.transform(image)
        return image, img_path

def detect_noise_in_folder(folder_path, model_path):
    device = get_device()
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train it first.")
        return

    # Load Model structure
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)
        
    num_ftrs = model.fc.in_features
    # We must match the class count from training (2 classes: clean, noisy)
    model.fc = nn.Linear(num_ftrs, 2) 
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model = model.to(device)
    model.eval()

    # Data config
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleImageDataset(folder_path, transform=transform)
    if len(dataset) == 0:
        print(f"No images found in {folder_path}")
        return
        
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print("-" * 80)
    print(f"{'Filename':<50} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 80)
    
    noisy_count = 0
    total_count = 0
    
    # Class mapping
    classes = ['Clean', 'NOISY'] 
    
    with torch.no_grad():
        for inputs, paths in tqdm(loader, unit="batch", ncols=100):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            # Get max probability class
            conf, preds = torch.max(probs, 1)
            
            for i in range(len(paths)):
                pred_idx = preds[i].item()
                confidence = conf[i].item()
                path = paths[i]
                filename = os.path.basename(path)
                
                label = classes[pred_idx]
                
                # Setup colors
                if label == 'NOISY':
                    color_start = "\033[91m" 
                    noisy_count += 1
                else:
                    color_start = "\033[92m"
                color_end = "\033[0m"
                
                print(f"{color_start}{filename:<50} | {label:<10} | {confidence:.2%}{color_end}")
                total_count += 1

    print("-" * 80)
    print(f"Comparison Result: {noisy_count}/{total_count} images flagged as NOISY.")

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
