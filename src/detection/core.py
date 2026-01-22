import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import cv2
import random
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from skimage.feature import peak_local_max
from torchvision import models, transforms
import torch.nn as nn

from src.detection.models import UNet, FeatureExtractor, RootTipDataset
from src.utils.common import get_device, discover_images, get_timestamp
from src.utils.logging import PerformanceLogger, calculate_pixel_accuracy

def detect_peaks(heatmap, threshold=0.5, min_distance=10):
    heatmap = heatmap.squeeze()
    coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold)
    return coords[:, [1, 0]]  # (x, y)

def get_tip_coords_pred(image, model, patch_size=512, stride=256, threshold=0.5, min_distance=10, batch_size=16):
    device = next(model.parameters()).device
    image = np.array(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    H, W = image.shape[:2]
    pred_heatmap = np.zeros((H, W), dtype=np.float32)
    
    patch_coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch_coords.append((x, y))
            
    # Process patches in batches
    for i in range(0, len(patch_coords), batch_size):
        batch_info = patch_coords[i:i + batch_size]
        
        # Batch tensor creation: use a single large tensor if possible
        patches = []
        for x, y in batch_info:
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
        
        # Stack all patches at once and convert to tensor
        batch_array = np.stack(patches).transpose(0, 3, 1, 2)
        batch_tensor = torch.from_numpy(batch_array).float().to(device) / 255.0
            
        # Use Mixed Precision (FP16) for faster inference with no noticeable loss in accuracy
        device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        with torch.amp.autocast(device_type):
            with torch.no_grad():
                preds = torch.sigmoid(model(batch_tensor))
        
        preds = preds.cpu().float().numpy() # Ensure float32 for numpy operations
            
        for idx, (x, y) in enumerate(batch_info):
            # Using bitwise OR for binary heatmaps might be faster if they were binary, 
            # but here they are floats from sigmoid, so maximum is correct for overlapping patches.
            pred_heatmap[y:y+patch_size, x:x+patch_size] = np.maximum(
                pred_heatmap[y:y+patch_size, x:x+patch_size], preds[idx, 0])

    coords = detect_peaks(pred_heatmap, threshold=threshold, min_distance=min_distance)
    tips = [[int(x), int(y), float(pred_heatmap[y, x])] for x, y in coords]
    return tips

def extract_features_from_patches(patches, extractor, device, batch_size=16):
    if not patches:
        return np.array([])
    
    # Process in batches
    features = []
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i + batch_size]
        if isinstance(batch[0], torch.Tensor):
            batch_tensor = torch.stack(batch).to(device)
        else:
            # Assume it's a list of numpy arrays
            batch_array = np.stack(batch).transpose(0, 3, 1, 2)
            batch_tensor = torch.from_numpy(batch_array).float().to(device) / 255.0
            
        with torch.no_grad():
            feat_maps = extractor(batch_tensor)
            fh, fw = feat_maps.shape[2:]
            cy, cx = fh // 2, fw // 2
            vecs = feat_maps[:, :, cy, cx].cpu().numpy()
            features.append(vecs)
    
    return np.concatenate(features, axis=0) if features else np.array([])

def export_features_for_folder(
    img_folder, 
    model_ckpt, 
    output_json,
    annotations_json=None,
    use_gt=False,
    extract_features=True,
    layer="enc3",
    threshold=0.5,
    log_file=None,
    filter_noise=False,
    noise_model_path=None):
    
    device = get_device()
    image_files = discover_images(img_folder)
    
    # --- Load Noise Classifier (if enabled) ---
    noise_model = None
    noise_transform = None
    if filter_noise and noise_model_path:
        print(f"Loading Noise Classifier from {noise_model_path}...")
        try:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                noise_model = models.resnet18(weights=weights)
            except AttributeError:
                noise_model = models.resnet18(pretrained=True)
            
            # Recreate the classification head (2 classes)
            num_ftrs = noise_model.fc.in_features
            noise_model.fc = nn.Linear(num_ftrs, 2)
            
            # Load weights
            noise_model.load_state_dict(torch.load(noise_model_path, map_location=device))
            noise_model.to(device)
            noise_model.eval()
            
            noise_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print("Noise Classifier loaded successfully.")
        except Exception as e:
            print(f"Error loading noise classifier: {e}")
            print("Proceeding without noise filtering.")
            noise_model = None
    
    base_model = UNet().to(device)
    base_model.load_state_dict(torch.load(model_ckpt, map_location=device))
    base_model.eval()
    
    extractor = None
    if extract_features:
        extractor = FeatureExtractor(base_model, layer).eval()
    
    ann = None
    if use_gt and annotations_json:
        with open(annotations_json) as f:
            ann = json.load(f)
    
    all_tips = []
    patch_size = 512
    half_patch = patch_size // 2
    
    # Initialize logger if log_file is provided
    logger = PerformanceLogger(log_file, log_type="detection") if log_file else None
    
    for img_path in tqdm(image_files, desc="Processing Images"):
        img_start_time = time.time()
        
        image = Image.open(img_path)
        image_arr = np.array(image)
        if image_arr.ndim == 2:
            image_arr = np.stack([image_arr] * 3, axis=-1)
        image_h, image_w = image_arr.shape[:2]
        
        # --- Noise Check ---
        if noise_model and noise_transform:
            try:
                img_tensor = noise_transform(image.convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = noise_model(img_tensor)
                    _, preds = torch.max(outputs, 1)
                    # Class 1 is NOISY, 0 is clean
                    is_noisy = (preds.item() == 1)
                    
                if is_noisy:
                    print(f"Skipping {img_path.name} (Detected as NOISY)")
                    if logger:
                        img_timestamp = get_timestamp(img_path.name)
                        # We don't log it, just ignore them for now.
                        
                        #logger.log_image_processing(
                        #    image_name=img_path.name,
                        #    num_tips=0,
                        #    processing_time=time.time() - img_start_time,
                        #    image_timestamp=img_timestamp
                        #)
                    continue # SKIP DETECTION
            except Exception as e:
                print(f"Error during noise check for {img_path.name}: {e}")
                # Fallthrough to detection if check fails? Or skip? 
                # Let's fallthrough to be safe, or print error.
                pass 
   
        if use_gt and ann:
            coords = ann.get(img_path.name, [])
            coords = [[x, y] for x, y in coords]
        else:
            coords = get_tip_coords_pred(image, base_model, threshold=float(threshold))
        
        if not coords:
            # Log even if no tips detected
            if logger:
                img_timestamp = get_timestamp(img_path.name)
                processing_time = time.time() - img_start_time
                logger.log_image_processing(
                    image_name=img_path.name,
                    num_tips=0,
                    processing_time=processing_time,
                    image_timestamp=img_timestamp
                )
            continue
            
        current_patches = []
        if extract_features:
            for coord in coords:
                cx, cy = coord[0], coord[1]
                x1 = int(max(0, cx - half_patch))
                y1 = int(max(0, cy - half_patch))
                x2 = int(min(image_w, x1 + patch_size))
                y2 = int(min(image_h, y1 + patch_size))
                
                if x2 - x1 < patch_size: x1 = max(0, x2 - patch_size)
                if y2 - y1 < patch_size: y1 = max(0, y2 - patch_size)
                
                patch = image_arr[y1:y1+patch_size, x1:x1+patch_size]
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    pad_h = patch_size - patch.shape[0]
                    pad_w = patch_size - patch.shape[1]
                    patch = np.pad(patch, ((0,pad_h),(0,pad_w),(0,0)), mode='constant')
                
                current_patches.append(patch) # Keep as numpy for now, stack in extract_features_from_patches

        if extract_features and extractor:
            features = extract_features_from_patches(current_patches, extractor, device)
        else:
            features = [None] * len(coords)
        
        for coord, feat in zip(coords, features):
            if len(coord) == 3:
                x, y, score = coord
            else:
                x, y = coord
                score = None if use_gt else float('nan')
            all_tips.append({
                "image": str(img_path.relative_to(img_folder)),
                "basename": img_path.name,
                "x": int(x),
                "y": int(y),
                "score": float(score) if score is not None else None,
                "features": feat.astype(float).tolist() if feat is not None else None
            })
        
        # Log processing metrics
        if logger:
            img_timestamp = get_timestamp(img_path.name)
            processing_time = time.time() - img_start_time
            logger.log_image_processing(
                image_name=img_path.name,
                num_tips=len(coords),
                processing_time=processing_time,
                image_timestamp=img_timestamp
            )

    # Ensure output directory exists
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as out:
        json.dump(all_tips, out, indent=2)
    print(f"Wrote {len(all_tips)} tip features to {output_json}")
    
    # Save performance log
    if logger:
        logger.save()

def train_detection(img_dir, ann_file, epochs=20, batch_size=4, val_split=0.2, patience=5, model_name="model.pth", log_file=None):
    device = get_device()
    with open(ann_file) as f:
        ann = json.load(f)
    
    all_images = discover_images(img_dir)
    random.shuffle(all_images)
    split_idx = int((1 - val_split) * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    train_set = RootTipDataset(train_images, ann)
    val_set = RootTipDataset(val_images, ann)
    
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    loss_fn = lambda pred, target: F.binary_cross_entropy_with_logits(pred, target)
    
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs("models", exist_ok=True)
    
    # Initialize logger if log_file is provided
    logger = PerformanceLogger(log_file, log_type="training_detection") if log_file else None
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if device == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred = model(x)
                    loss = loss_fn(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                
            total_train_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(pred)
                accuracy = calculate_pixel_accuracy(pred_sigmoid, y)
                total_train_accuracy += accuracy
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1} [{i}/{len(loader)}] Loss: {loss.item():.6f} Acc: {accuracy:.4f}")

        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss = loss_fn(pred, y)
                total_val_loss += val_loss.item()
                
                # Calculate validation accuracy
                pred_sigmoid = torch.sigmoid(pred)
                val_accuracy = calculate_pixel_accuracy(pred_sigmoid, y)
                total_val_accuracy += val_accuracy
        
        avg_train_loss = total_train_loss / len(loader)
        avg_train_accuracy = total_train_accuracy / len(loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_val_accuracy / len(val_loader)
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.6f} Train Acc: {avg_train_accuracy:.4f}")
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.6f} Val Acc: {avg_val_accuracy:.4f} Time: {epoch_time:.2f}s")
        
        # Log metrics
        if logger:
            logger.log_training_epoch(
                epoch=epoch+1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                train_accuracy=avg_train_accuracy,
                val_accuracy=avg_val_accuracy,
                epoch_time=epoch_time
            )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join("models", model_name))
            print(f"Model saved to models/{model_name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save performance log
    if logger:
        logger.save()
