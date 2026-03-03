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
from src.utils.common import get_device, get_timestamp, get_plant_id, filter_outlier_images, get_time_diff_hours
from src.utils.logging import PerformanceLogger

def compute_cost_matrix_from_features(tips1, tips2, model, device, spatial_threshold=150, time_diff=1.0):
    num1 = len(tips1)
    num2 = len(tips2)
    cost_matrix = np.full((num1, num2), 1e6)

    if num1 == 0 or num2 == 0:
        return cost_matrix

    # 1. Prepare features and coordinates as tensors
    if hasattr(tips1[0], '_feat_tensor') and tips1[0]['_feat_tensor'] is not None:
        feats1 = torch.stack([t['_feat_tensor'] for t in tips1])
    else:
        feats1 = torch.tensor([t['features'] for t in tips1], device=device).float()
    
    coords1 = torch.tensor([[t['x'], t['y']] for t in tips1], device=device).float() # (N1, 2)
    
    if hasattr(tips2[0], '_feat_tensor') and tips2[0]['_feat_tensor'] is not None:
        feats2 = torch.stack([t['_feat_tensor'] for t in tips2])
    else:
        feats2 = torch.tensor([t['features'] for t in tips2], device=device).float()
    
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
    
    dxy = (coords2[idx2] - coords1[idx1]) / max(0.01, time_diff)
    dist_batch = (dist_matrix[idx1, idx2].unsqueeze(1)) / max(0.01, time_diff)
    
    spatial_feats = torch.cat([dxy / 100.0, dist_batch / 100.0], dim=1) 
    batch_input = torch.cat([f1_batch, f2_batch, spatial_feats], dim=1) 
    
    # 5. Run model inference
    with torch.no_grad():
        probs = model(batch_input).squeeze(1).cpu().numpy() 
        
    # 6. Fill the cost matrix
    cost_matrix[idx1.cpu().numpy(), idx2.cpu().numpy()] = 1.0 - probs
    
    return cost_matrix

def compute_gnn_cost_matrix(tips1, tips2, model, device, spatial_threshold=150, time_diff=1.0):
    """
    Computes the cost (1 - probability) matrix using the GNN model.
    """
    num1 = len(tips1)
    num2 = len(tips2)
    cost_matrix = np.full((num1, num2), 1e6) 
    
    if num1 == 0 or num2 == 0:
        return cost_matrix
        
    if hasattr(tips1[0], '_feat_tensor') and tips1[0]['_feat_tensor'] is not None:
        feat1 = torch.stack([t['_feat_tensor'] for t in tips1])
    else:
        feat1 = torch.tensor([t['features'] for t in tips1], device=device).float()
    coord1 = torch.tensor([[t['x'], t['y']] for t in tips1], device=device).float()
    
    if hasattr(tips2[0], '_feat_tensor') and tips2[0]['_feat_tensor'] is not None:
        feat2 = torch.stack([t['_feat_tensor'] for t in tips2])
    else:
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
    
    t_delta = max(0.01, time_diff)
    dxy = (coord2[dst_local] - coord1[src_local]) / t_delta
    dist = (dist_matrix[src_local, dst_local].unsqueeze(1)) / t_delta
    edge_spatials = torch.cat([dxy, dist], dim=1) / 100.0
    
    with torch.no_grad():
        logits = model(all_feats, edge_index, edge_spatials)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        
    cost_matrix[src_local.cpu().numpy(), dst_local.cpu().numpy()] = 1.0 - probs
    
    return cost_matrix

def associate_tips_multi_plant(features_json, model_path, output_json, output_links_json=None, spatial_threshold=150, prob_threshold=0.2, log_file=None, stop_event=None):
    device = get_device()
    
    with open(features_json, 'r') as f:
        raw_data = json.load(f)
        
    # Group by plant ID, then by image
    plant_data = {}
    for entry in raw_data:
        pid = get_plant_id(entry.get('basename', os.path.basename(entry['image'])))
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
        
        # --- OPTIMIZATION: Pre-convert all features to tensors for this plant ---
        for img_rel in sorted_imgs:
            for t in images_dict[img_rel]:
                # Pre-convert and store on device to avoid redundant copies
                t['_feat_tensor'] = torch.tensor(t['features'], device=device).float()

        prev_tips = []

        for img_rel_path in tqdm(sorted_imgs, desc=f"Tracking {pid}"):
            if stop_event and stop_event.is_set():
                print("Tracking cancelled by user.")
                return
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
                    "_feat_tensor": t.get('_feat_tensor'), # Pass the pre-converted tensor
                    "id": -1,
                    "image": t['image'], # full relative path
                    "basename": t.get('basename', os.path.basename(t['image']))
                })

            if not prev_tips:
                for t in curr_tips:
                    t["id"] = global_next_id
                    global_next_id += 1
            else:
                # Calculate time difference for normalization
                t1_name = prev_tips[0]['basename']
                t2_name = curr_tips[0]['basename']
                time_diff = get_time_diff_hours(t1_name, t2_name)
                
                if is_gnn:
                    costs = compute_gnn_cost_matrix(prev_tips, curr_tips, model, device, spatial_threshold, time_diff)
                else:
                    costs = compute_cost_matrix_from_features(prev_tips, curr_tips, model, device, spatial_threshold, time_diff)
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
                "basename": curr_tips[0].get("basename", os.path.basename(curr_tips[0]["image"])),
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

def _prepare_data_from_consolidated_links(links_json):
    """
    Reconstructs all_features and all_links structures from the consolidated JSON.
    """
    with open(links_json, 'r') as f:
        formatted_links = json.load(f)
        
    all_features = []
    all_links_index_based = {}
    
    # Track which tips we've already added to all_features to avoid duplicates
    # Key: (image_basename, tip_x, tip_y)
    seen_tips = {}
    
    for pair_key, pairs in formatted_links.items():
        if "->" not in pair_key:
            continue
            
        img1_name, img2_name = pair_key.split("->")
        all_links_index_based[pair_key] = []
        
        for p in pairs:
            # Reconstruct tip1
            t1_key = (img1_name, p['tip1_x'], p['tip1_y'])
            if t1_key not in seen_tips:
                t1 = {
                    'image': img1_name,
                    'basename': img1_name,
                    'x': p['tip1_x'],
                    'y': p['tip1_y'],
                    'features': p.get('tip1_features', [])
                }
                seen_tips[t1_key] = len([f for f in all_features if f['basename'] == img1_name])
                all_features.append(t1)
            
            # Reconstruct tip2
            t2_key = (img2_name, p['tip2_x'], p['tip2_y'])
            if t2_key not in seen_tips:
                t2 = {
                    'image': img2_name,
                    'basename': img2_name,
                    'x': p['tip2_x'],
                    'y': p['tip2_y'],
                    'features': p.get('tip2_features', [])
                }
                seen_tips[t2_key] = len([f for f in all_features if f['basename'] == img2_name])
                all_features.append(t2)
                
            # Note: The indices in the formatted_links were relative to the tips list 
            # for that image AT THE TIME OF ANNOTATION.
            # However, resolve_link_indices handles matching via coordinates, 
            # which is preferred and robust.
            all_links_index_based[pair_key].append({
                'tip1_index': p['tip1_index'],
                'tip2_index': p['tip2_index'],
                'tip1_x': p['tip1_x'],
                'tip1_y': p['tip1_y'],
                'tip2_x': p['tip2_x'],
                'tip2_y': p['tip2_y']
            })
            
    return all_features, all_links_index_based

def train_linker(links_json, epochs=20, batch_size=32, model_name="tip_linker.pth", log_file=None, use_gnn=True, stop_event=None):
    device = get_device()
    
    print(f"Loading training data from consolidated file: {links_json}")
    all_features, all_links = _prepare_data_from_consolidated_links(links_json)
        
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", model_name)
    logger = PerformanceLogger(log_file, log_type="training_linker") if log_file else None

    if use_gnn:
        # GNN Training Logic (from train_gnn_standalone.py)
        # Organize features by basename
        features_by_img = {}
        for item in all_features:
            bname = item.get('basename', os.path.basename(item['image']))
            if bname not in features_by_img:
                features_by_img[bname] = []
            features_by_img[bname].append(item)
            
        train_pairs = []
        for link_key, links in all_links.items():
            img1_name, img2_name = link_key.split("->")
            if img1_name in features_by_img and img2_name in features_by_img:
                from src.association.gnn_utils import resolve_link_indices
                true_set = resolve_link_indices(links, features_by_img[img1_name], features_by_img[img2_name])
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
        # Weighting: User prefers missing a link over creating a false positive.
        # Reduced from 50.0 to 1.0 to reflect this bias.
        pos_weight = torch.tensor([1.0], device=device) 
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Stability: Learning rate scheduler and best model tracking
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        best_val_loss = float('inf')

        from src.association.gnn_utils import build_graph_batch, resolve_link_indices
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # --- Training Stage ---
            model.train()
            total_train_loss = 0
            train_count = 0
            np.random.shuffle(train_set)
            
            for pair in train_set:
                if stop_event and stop_event.is_set():
                    print("Training cancelled by user.")
                    return
                t1_name = pair['img1'][0].get('basename', os.path.basename(pair['img1'][0]['image']))
                t2_name = pair['img2'][0].get('basename', os.path.basename(pair['img2'][0]['image']))
                time_diff = get_time_diff_hours(t1_name, t2_name)
                
                batch_data = build_graph_batch(pair['img1'], pair['img2'], pair['links'], device, time_diff=time_diff)
                if batch_data is None: continue
                
                node_feats, edge_index, edge_spatials, labels = batch_data
                optimizer.zero_grad()
                preds = model(node_feats, edge_index, edge_spatials)
                loss = criterion(preds, labels)
                loss.backward()
                
                # Stability: Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_train_loss += loss.item()
                train_count += 1
            
            avg_train_loss = total_train_loss / max(1, train_count)
            
            # --- Validation Stage ---
            model.eval()
            total_val_loss = 0
            val_count = 0
            with torch.no_grad():
                for pair in val_set:
                    t1_name = pair['img1'][0].get('basename', os.path.basename(pair['img1'][0]['image']))
                    t2_name = pair['img2'][0].get('basename', os.path.basename(pair['img2'][0]['image']))
                    time_diff = get_time_diff_hours(t1_name, t2_name)
                    
                    batch_data = build_graph_batch(pair['img1'], pair['img2'], pair['links'], device, time_diff=time_diff)
                    if batch_data is None: continue
                    node_feats, edge_index, edge_spatials, labels = batch_data
                    preds = model(node_feats, edge_index, edge_spatials)
                    loss = criterion(preds, labels)
                    total_val_loss += loss.item()
                    val_count += 1
            avg_val_loss = total_val_loss / max(1, val_count)
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} GNN Loss - Train: {avg_train_loss:.6f} Val: {avg_val_loss:.6f} Time: {epoch_time:.2f}s")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"  --> Saved new best model with val_loss: {avg_val_loss:.6f}")
            
            if logger:
                logger.log_training_epoch(epoch=epoch+1, train_loss=avg_train_loss, val_loss=avg_val_loss, epoch_time=epoch_time)
        
    else:
        # Original MLP Training Logic
        tip_features = {}
        for f in all_features:
            b = f.get('basename', os.path.basename(f['image']))
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
                if stop_event and stop_event.is_set():
                    print("Training cancelled by user.")
                    return
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
