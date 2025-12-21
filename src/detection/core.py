import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import cv2
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from skimage.feature import peak_local_max

from src.detection.models import UNet, FeatureExtractor, RootTipDataset
from src.utils.common import get_device, discover_images

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
            
        with torch.no_grad():
            preds = torch.sigmoid(model(batch_tensor)).cpu().numpy()
            
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
    threshold=0.5):
    
    device = get_device()
    image_files = discover_images(img_folder)
    
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
    
    for img_path in tqdm(image_files, desc="Processing Images"):
        image = Image.open(img_path)
        image_arr = np.array(image)
        if image_arr.ndim == 2:
            image_arr = np.stack([image_arr] * 3, axis=-1)
        image_h, image_w = image_arr.shape[:2]
   
        if use_gt and ann:
            coords = ann.get(img_path.name, [])
            coords = [[x, y] for x, y in coords]
        else:
            coords = get_tip_coords_pred(image, base_model, threshold=float(threshold))
        
        if not coords:
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

    # Ensure output directory exists
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as out:
        json.dump(all_tips, out, indent=2)
    print(f"Wrote {len(all_tips)} tip features to {output_json}")

def train_detection(img_dir, ann_file, epochs=20, batch_size=4, val_split=0.2, patience=5, model_name="model.pth"):
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
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
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
            if i % 10 == 0:
                print(f"Epoch {epoch+1} [{i}/{len(loader)}] Loss: {loss.item():.6f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss = loss_fn(pred, y)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join("models", model_name))
            print(f"Model saved to models/{model_name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
