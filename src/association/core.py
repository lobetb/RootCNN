import torch
import json
import numpy as np
import os
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.association.models import AffinityMLP, LinkingDataset
from src.association.gnn_models import TipGNN
from src.utils.common import get_device, get_timestamp, get_plant_id, filter_outlier_images
from src.utils.logging import PerformanceLogger

def compute_cost_matrix_from_features(tips1, tips2, model, device, spatial_threshold=150):
    num1 = len(tips1)
    num2 = len(tips2)
    cost_matrix = np.full((num1, num2), 1e6)

    if num1 == 0 or num2 == 0:
        return cost_matrix

    # 1. Prepare features and coordinates as tensors
    feats1 = torch.tensor([t['features'] for t in tips1], device=device).float() # (N1, 256)
    coords1 = torch.tensor([[t['x'], t['y']] for t in tips1], device=device).float() # (N1, 2)
    
    feats2 = torch.tensor([t['features'] for t in tips2], device=device).float() # (N2, 256)
    coords2 = torch.tensor([[t['x'], t['y']] for t in tips2], device=device).float() # (N2, 2)

    # 2. Compute pairwise distances
    dist_matrix = torch.cdist(coords1, coords2, p=2) 
    
    # 3. Filter pairs by spatial threshold
    mask = dist_matrix <= spatial_threshold
    idx1, idx2 = torch.where(mask)
    
    if idx1.numel() == 0:
        return cost_matrix

    # 4. Prepare batch inputs
    f1_batch = feats1[idx1] 
    f2_batch = feats2[idx2] 
    
    dxy = coords2[idx2] - coords1[idx1] 
    dist_batch = dist_matrix[idx1, idx2].unsqueeze(1) 
    
    spatial_feats = torch.cat([dxy / 100.0, dist_batch / 100.0], dim=1) 
    batch_input = torch.cat([f1_batch, f2_batch, spatial_feats], dim=1) 
    
    # 5. Run model inference
    with torch.no_grad():
        probs = model(batch_input).squeeze(1).cpu().numpy() 
        
    # 6. Fill the cost matrix
    cost_matrix[idx1.cpu().numpy(), idx2.cpu().numpy()] = 1.0 - probs
    
    return cost_matrix

def compute_gnn_cost_matrix(tips1, tips2, model, device, spatial_threshold=150):
    """
    Computes the cost (1 - probability) matrix using the GNN model.
    """
    num1 = len(tips1)
    num2 = len(tips2)
    cost_matrix = np.full((num1, num2), 1e6) 
    
    if num1 == 0 or num2 == 0:
        return cost_matrix
        
    feat1 = torch.tensor([t['features'] for t in tips1], device=device).float()
    coord1 = torch.tensor([[t['x'], t['y']] for t in tips1], device=device).float()
    
    feat2 = torch.tensor([t['features'] for t in tips2], device=device).float()
    coord2 = torch.tensor([[t['x'], t['y']] for t in tips2], device=device).float()
    
    # 1. Build Graph
    all_feats = torch.cat([feat1, feat2], dim=0)
    
    dist_matrix = torch.cdist(coord1, coord2)
    mask = dist_matrix < spatial_threshold # Only consider plausible edges
    
    src_local, dst_local = torch.where(mask)
    if src_local.numel() == 0:
        return cost_matrix
        
    src_global = src_local
    dst_global = dst_local + num1
    
    edge_index = torch.stack([src_global, dst_global], dim=0)
    
    dxy = coord2[dst_local] - coord1[src_local]
    dist = dist_matrix[src_local, dst_local].unsqueeze(1)
    edge_spatials = torch.cat([dxy, dist], dim=1) / 100.0
    
    with torch.no_grad():
        logits = model(all_feats, edge_index, edge_spatials)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        
    cost_matrix[src_local.cpu().numpy(), dst_local.cpu().numpy()] = 1.0 - probs
    
    return cost_matrix

def associate_tips_multi_plant(features_json, model_path, output_json, output_links_json=None, spatial_threshold=150, prob_threshold=0.2, log_file=None):
    device = get_device()
    
    with open(features_json, 'r') as f:
        raw_data = json.load(f)
        
    # Group by plant ID, then by image
    plant_data = {}
    for entry in raw_data:
        pid = get_plant_id(entry['basename'])
        if pid not in plant_data:
            plant_data[pid] = {}
        
        # Group by plant ID, then by image relative path
        img_rel_path = entry['image']
        if img_rel_path not in plant_data[pid]:
            plant_data[pid][img_rel_path] = []
        plant_data[pid][img_rel_path].append(entry)
        
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine which model to use based on the checkpoint
    # TipGNN has 'node_encoder' while AffinityMLP doesn't
    if any("node_encoder" in k for k in checkpoint.keys()):
        model = TipGNN().to(device)
        is_gnn = True
    else:
        model = AffinityMLP().to(device)
        is_gnn = False
        
    model.load_state_dict(checkpoint)
    model.eval()

    all_tracks = []
    all_positive_links = []
    global_next_id = 0
    
    # Initialize logger if log_file is provided
    logger = PerformanceLogger(log_file, log_type="tracking") if log_file else None

    for pid, images_dict in plant_data.items():
        print(f"Processing Plant ID: {pid}")
        sorted_imgs = sorted(images_dict.keys(), key=get_timestamp)
        valid_imgs = filter_outlier_images(sorted_imgs, images_dict)
        
        prev_tips = []
        for img_rel_path in tqdm(sorted_imgs, desc=f"Tracking {pid}"):
            img_start_time = time.time()
            if img_rel_path not in valid_imgs:
                continue
            curr_tips_raw = images_dict[img_rel_path]
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
                if is_gnn:
                    costs = compute_gnn_cost_matrix(prev_tips, curr_tips, model, device, spatial_threshold)
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
                "basename": curr_tips[0]["basename"],
                "plant_id": pid,
                "tips": [{"id": t["id"], "x": t["x"], "y": t["y"], "score": t["score"], "assoc_score": t.get("assoc_score", 0.0)} for t in curr_tips]
            })
            
            # Log tracking metrics
            if logger:
                img_timestamp = get_timestamp(img_rel_path)
                processing_time = time.time() - img_start_time
                num_associations = len([t for t in curr_tips if t.get("assoc_score", 0.0) > 0])
                logger.log_image_processing(
                    image_name=img_rel_path,
                    num_tips=len(curr_tips),
                    processing_time=processing_time,
                    image_timestamp=img_timestamp,
                    num_associations=num_associations,
                    additional_metrics={"plant_id": pid}
                )
            
            prev_tips = curr_tips

    with open(output_json, 'w') as f:
        json.dump(all_tracks, f, indent=2)
    print(f"Saved {len(all_tracks)} frames of tracks to {output_json}")

    if output_links_json:
        with open(output_links_json, 'w') as f:
            json.dump(all_positive_links, f, indent=2)
        print(f"Saved {len(all_positive_links)} positive associations to {output_links_json}")
    
    # Save performance log
    if logger:
        logger.save()

def train_linker(features_json, links_json, epochs=20, batch_size=32, model_name="tip_linker.pth", log_file=None, use_gnn=True):
    device = get_device()
    
    with open(features_json, 'r') as f:
        all_features = json.load(f)
    with open(links_json, 'r') as f:
        all_links = json.load(f)
        
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", model_name)
    logger = PerformanceLogger(log_file, log_type="training_linker") if log_file else None

    if use_gnn:
        # GNN Training Logic (from train_gnn_standalone.py)
        # Organize features by basename
        features_by_img = {}
        for item in all_features:
            if item['basename'] not in features_by_img:
                features_by_img[item['basename']] = []
            features_by_img[item['basename']].append(item)
            
        train_pairs = []
        for link_key, links in all_links.items():
            img1_name, img2_name = link_key.split("->")
            if img1_name in features_by_img and img2_name in features_by_img:
                true_set = set((l['tip1_index'], l['tip2_index']) for l in links)
                train_pairs.append({
                    'img1': features_by_img[img1_name],
                    'img2': features_by_img[img2_name],
                    'links': true_set
                })
        
        np.random.shuffle(train_pairs)
        split_idx = int(len(train_pairs) * 0.9)
        val_set = train_pairs[split_idx:]
        train_set = train_pairs[:split_idx]
        
        model = TipGNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        pos_weight = torch.tensor([50.0], device=device) 
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        from train_gnn_standalone import build_graph_batch
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            model.train()
            total_loss = 0
            count = 0
            np.random.shuffle(train_set)
            
            for pair in train_set:
                batch_data = build_graph_batch(pair['img1'], pair['img2'], pair['links'], device)
                if batch_data is None: continue
                
                node_feats, edge_index, edge_spatials, labels = batch_data
                optimizer.zero_grad()
                preds = model(node_feats, edge_index, edge_spatials)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            
            avg_loss = total_loss / max(1, count)
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} GNN Loss: {avg_loss:.6f} Time: {epoch_time:.2f}s")
            
            if logger:
                logger.log_training_epoch(epoch=epoch+1, train_loss=avg_loss, val_loss=avg_loss, epoch_time=epoch_time)
                
        torch.save(model.state_dict(), model_save_path)
        
    else:
        # Original MLP Training Logic
        tip_features = {}
        for f in all_features:
            b = f['basename']
            if b not in tip_features: tip_features[b] = []
            tip_features[b].append(f)
            
        dataset = LinkingDataset(tip_features, all_links)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = AffinityMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCELoss()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
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
            
            avg_loss = total_loss / len(loader)
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} MLP Loss: {avg_loss:.6f} Time: {epoch_time:.2f}s")
            
            if logger:
                logger.log_training_epoch(epoch=epoch+1, train_loss=avg_loss, val_loss=avg_loss, epoch_time=epoch_time)
        
        torch.save(model.state_dict(), model_save_path)

    print(f"Linker model saved to {model_save_path}")
    if logger: logger.save()
