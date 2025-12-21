import torch
import json
import numpy as np
import os
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.association.models import AffinityMLP, LinkingDataset
from src.utils.common import get_device, get_timestamp, get_plant_id, filter_outlier_images

def compute_cost_matrix_from_features(tips1, tips2, model, device, spatial_threshold=150):
    num1 = len(tips1)
    num2 = len(tips2)
    cost_matrix = np.full((num1, num2), 1e6)

    if num1 == 0 or num2 == 0:
        return cost_matrix

    # 1. Prepare features and coordinates as tensors
    # Moving all data to the device at once is much faster than per-tip conversions
    feats1 = torch.tensor([t['features'] for t in tips1], device=device).float() # (N1, 256)
    coords1 = torch.tensor([[t['x'], t['y']] for t in tips1], device=device).float() # (N1, 2)
    
    feats2 = torch.tensor([t['features'] for t in tips2], device=device).float() # (N2, 256)
    coords2 = torch.tensor([[t['x'], t['y']] for t in tips2], device=device).float() # (N2, 2)

    # 2. Compute pairwise distances using torch.cdist (very fast on GPU)
    # Output shape: (N1, N2)
    dist_matrix = torch.cdist(coords1, coords2, p=2) 
    
    # 3. Filter pairs by spatial threshold
    mask = dist_matrix <= spatial_threshold
    idx1, idx2 = torch.where(mask)
    
    if idx1.numel() == 0:
        return cost_matrix

    # 4. Prepare batch inputs efficiently
    # We use indexing to create the batch of pairs without any Python loops
    f1_batch = feats1[idx1] # (K, 256)
    f2_batch = feats2[idx2] # (K, 256)
    
    dxy = coords2[idx2] - coords1[idx1] # (K, 2)
    dist_batch = dist_matrix[idx1, idx2].unsqueeze(1) # (K, 1)
    
    # Concatenate [feat1, feat2, dx/100, dy/100, dist/100]
    spatial_feats = torch.cat([dxy / 100.0, dist_batch / 100.0], dim=1) # (K, 3)
    batch_input = torch.cat([f1_batch, f2_batch, spatial_feats], dim=1) # (K, 515)
    
    # 5. Run model inference on the whole batch
    with torch.no_grad():
        probs = model(batch_input).squeeze(1).cpu().numpy() # (K,)
        
    # 6. Efficiently fill the cost matrix
    cost_matrix[idx1.cpu().numpy(), idx2.cpu().numpy()] = 1.0 - probs
    
    return cost_matrix

def associate_tips_multi_plant(features_json, model_path, output_json, output_links_json=None, spatial_threshold=150, prob_threshold=0.2):
    device = get_device()
    
    with open(features_json, 'r') as f:
        raw_data = json.load(f)
        
    # Group by plant ID, then by image
    plant_data = {}
    for entry in raw_data:
        pid = get_plant_id(entry['basename'])
        if pid not in plant_data:
            plant_data[pid] = {}
        
        img_name = entry['basename']
        if img_name not in plant_data[pid]:
            plant_data[pid][img_name] = []
        plant_data[pid][img_name].append(entry)
        
    model = AffinityMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_tracks = []
    all_positive_links = []
    global_next_id = 0

    for pid, images_dict in plant_data.items():
        print(f"Processing Plant ID: {pid}")
        sorted_imgs = sorted(images_dict.keys(), key=get_timestamp)
        valid_imgs = filter_outlier_images(sorted_imgs, images_dict)
        
        prev_tips = []
        for img_name in tqdm(sorted_imgs, desc=f"Tracking {pid}"):
            if img_name not in valid_imgs:
                continue
            curr_tips_raw = images_dict[img_name]
            curr_tips = []
            for t in curr_tips_raw:
                curr_tips.append({
                    "x": t['x'],
                    "y": t['y'],
                    "score": t.get('score', 1.0),
                    "features": t['features'],
                    "id": -1,
                    "image": t['image'], # full relative path
                    "basename": t['basename']
                })

            if not prev_tips:
                for t in curr_tips:
                    t["id"] = global_next_id
                    global_next_id += 1
            else:
                costs = compute_cost_matrix_from_features(prev_tips, curr_tips, model, device, spatial_threshold)
                row_ind, col_ind = linear_sum_assignment(costs)
                
                assigned_curr = set()
                for r, c in zip(row_ind, col_ind):
                    prob = 1.0 - costs[r, c]
                    if prob >= prob_threshold:
                        curr_tips[c]["id"] = prev_tips[r]["id"]
                        curr_tips[c]["assoc_score"] = float(prob)
                        assigned_curr.add(c)
                        
                        # Add to positive links
                        c1 = np.array([prev_tips[r]['x'], prev_tips[r]['y']])
                        c2 = np.array([curr_tips[c]['x'], curr_tips[c]['y']])
                        dist = np.linalg.norm(c1 - c2)
                        all_positive_links.append({
                            "tip1": {
                                "image": prev_tips[r]["image"],
                                "x": int(prev_tips[r]["x"]),
                                "y": int(prev_tips[r]["y"]),
                                "id": int(prev_tips[r]["id"])
                            },
                            "tip2": {
                                "image": curr_tips[c]["image"],
                                "x": int(curr_tips[c]["x"]),
                                "y": int(curr_tips[c]["y"]),
                                "id": int(curr_tips[c]["id"])
                            },
                            "distance_px": float(dist)
                        })
                
                for i in range(len(curr_tips)):
                    if i not in assigned_curr:
                        curr_tips[i]["id"] = global_next_id
                        curr_tips[i]["assoc_score"] = 0.0
                        global_next_id += 1
            
            all_tracks.append({
                "image": curr_tips[0]["image"],
                "basename": img_name,
                "plant_id": pid,
                "tips": [{"id": t["id"], "x": t["x"], "y": t["y"], "score": t["score"], "assoc_score": t.get("assoc_score", 0.0)} for t in curr_tips]
            })
            prev_tips = curr_tips

    with open(output_json, 'w') as f:
        json.dump(all_tracks, f, indent=2)
    print(f"Saved {len(all_tracks)} frames of tracks to {output_json}")

    if output_links_json:
        with open(output_links_json, 'w') as f:
            json.dump(all_positive_links, f, indent=2)
        print(f"Saved {len(all_positive_links)} positive associations to {output_links_json}")

def train_linker(features_json, links_json, epochs=20, batch_size=32, model_name="tip_linker.pth"):
    device = get_device()
    
    with open(features_json, 'r') as f:
        features_list = json.load(f)
    
    # tip_features mapping: basename -> list of tips
    tip_features = {}
    for f in features_list:
        b = f['basename']
        if b not in tip_features:
            tip_features[b] = []
        tip_features[b].append(f)
        
    with open(links_json, 'r') as f:
        tip_links = json.load(f)
        
    dataset = LinkingDataset(tip_features, tip_links)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AffinityMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()
    
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} Linker Loss: {total_loss/len(loader):.6f}")
        
    torch.save(model.state_dict(), os.path.join("models", model_name))
    print(f"Linker model saved to models/{model_name}")
