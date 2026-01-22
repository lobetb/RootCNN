import torch
import numpy as np
import json
import time
import os
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from src.utils.common import get_device, get_timestamp, get_plant_id, filter_outlier_images
from src.utils.logging import PerformanceLogger

def compute_cost_matrix_heuristic(tips1, tips2, device, w_spatial=1.0, w_feature=1.0, max_dist=150):
    """
    Computes a cost matrix based on spatial distance and feature cosine distance.
    
    Args:
        tips1: List of tip dicts from frame t-1
        tips2: List of tip dicts from frame t
        device: torch device
        w_spatial: Weight for spatial distance component
        w_feature: Weight for feature distance component
        max_dist: Maximum pixel distance allowed for a valid link
        
    Returns:
        cost_matrix: (N1, N2) numpy array. Values > 1e5 indicate invalid links.
    """
    num1 = len(tips1)
    num2 = len(tips2)
    cost_matrix = np.full((num1, num2), 1e6)

    if num1 == 0 or num2 == 0:
        return cost_matrix

    # Prepare data
    feats1 = torch.tensor([t['features'] for t in tips1], device=device).float() # (N1, 256)
    coords1 = torch.tensor([[t['x'], t['y']] for t in tips1], device=device).float() # (N1, 2)
    
    feats2 = torch.tensor([t['features'] for t in tips2], device=device).float() # (N2, 256)
    coords2 = torch.tensor([[t['x'], t['y']] for t in tips2], device=device).float() # (N2, 2)

    # 1. Spatial Distance (Euclidean)
    # (N1, N2)
    dist_matrix = torch.cdist(coords1, coords2, p=2)
    
    # 2. Feature Distance (Cosine Distance = 1 - Cosine Similarity)
    # Normalize features first
    f1_norm = torch.nn.functional.normalize(feats1, p=2, dim=1)
    f2_norm = torch.nn.functional.normalize(feats2, p=2, dim=1)
    # similarity = f1 . f2^T (N1, N2)
    sim_matrix = torch.mm(f1_norm, f2_norm.t())
    feat_dist_matrix = 1.0 - sim_matrix

    # 3. Combine Costs
    # Normalize spatial distance to be roughly comparable. 
    # E.g., max_dist=150. we can divide by 100 or simply use the raw value if weight is small.
    # Let's use the raw value but weighted.
    # A common scale: 10 pixels ~= 0.1 feature dist? 
    # Usually feature dist is 0.0 to 2.0 (cosine). 
    # Spatial dist can be 0 to 150+.
    # Let's normalize spatial by max_dist for the cost calculation to keep them in [0, 1] range roughly?
    # Or just rely on weights. Let's rely on weights but normalize spatial locally effectively.
    
    # Mask out too far pairs
    mask = dist_matrix <= max_dist
    
    # Calculate combined cost only where mask is true (optimization)
    # However, since we do matrix ops, we can just compute all and mask at the end.
    
    # Cost = w_s * (dist / max_dist) + w_f * feat_dist
    # If dist > max_dist, we will set to infinity later.
    
    normalized_spatial = dist_matrix / max_dist
    total_cost = w_spatial * normalized_spatial + w_feature * feat_dist_matrix
    
    # Apply mask
    # We set invalid ones to a very high cost
    total_cost[~mask] = 1e6
    
    return total_cost.cpu().numpy()

def associate_tips_heuristic(
    features_json, 
    output_json, 
    output_links_json=None, 
    spatial_weight=1.0, 
    feature_weight=1.0, 
    max_dist=150,
    prob_threshold=0.5, # Interpretation depends on cost, here used as a cutoff for valid assignment?
                        # Actually for linear assignment we minimize cost. 
                        # We need a threshold to reject assignments that are "too expensive".
    log_file=None
):
    """
    Standalone function to associate tips using the heuristic linker.
    """
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

    all_tracks = []
    all_positive_links = []
    global_next_id = 0
    
    logger = PerformanceLogger(log_file, log_type="tracking_heuristic") if log_file else None

    # Max cost allowed for an association to be considered valid
    # Since we normalize spatial to [0,1] at max_dist (approx), and feature to [0, 2]
    # Max reasonable cost might be around w_spatial * 1 + w_feature * 0.5 ??
    # A simple threshold is hard. Let's just say if cost < 1e5 (which checks max_dist) AND 
    # maybe we want strict feature matching?
    # For now, we rely on max_dist. The linear assignment finds the globally optimal.
    # We can add a post-check: if assigned cost is too high, drop it.
    max_valid_cost = spatial_weight * 1.0 + feature_weight * 2.0 # Upper bound theoretical
    
    # Let's say we reject if cost > threshold
    # user passed prob_threshold, but that was for MLP (0 to 1 probability).
    # Here we have cost (0 to infinity).
    # Let's define cost_threshold.
    cost_threshold = 1000.0 # Effectively only max_dist filters for now unless we tune this.

    for pid, images_dict in plant_data.items():
        print(f"Processing Plant ID: {pid}")
        sorted_imgs = sorted(images_dict.keys(), key=get_timestamp)
        valid_imgs = filter_outlier_images(sorted_imgs, images_dict)
        
        prev_tips = []
        for img_name in tqdm(sorted_imgs, desc=f"Tracking {pid}"):
            img_start_time = time.time()
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
                    "image": t['image'], 
                    "basename": t['basename']
                })
            
            if not prev_tips:
                for t in curr_tips:
                    t["id"] = global_next_id
                    global_next_id += 1
            else:
                costs = compute_cost_matrix_heuristic(
                    prev_tips, curr_tips, device, 
                    w_spatial=spatial_weight, 
                    w_feature=feature_weight, 
                    max_dist=max_dist
                )
                
                row_ind, col_ind = linear_sum_assignment(costs)
                
                assigned_curr = set()
                for r, c in zip(row_ind, col_ind):
                    cost_val = costs[r, c]
                    
                    # Check validity
                    if cost_val < 1e5: # Basic check (valid link)
                        # We can enforce stricter threshold here if needed
                        curr_tips[c]["id"] = prev_tips[r]["id"]
                        curr_tips[c]["assoc_cost"] = float(cost_val)
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
                            "distance_px": float(dist),
                            "cost": float(cost_val)
                        })
                
                # Assign new IDs to unassigned tips
                for i in range(len(curr_tips)):
                    if i not in assigned_curr:
                        curr_tips[i]["id"] = global_next_id
                        curr_tips[i]["assoc_cost"] = -1.0
                        global_next_id += 1
            
            all_tracks.append({
                "image": curr_tips[0]["image"],
                "basename": img_name,
                "plant_id": pid,
                "tips": [{"id": t["id"], "x": t["x"], "y": t["y"], "score": t["score"], "assoc_cost": t.get("assoc_cost", -1.0)} for t in curr_tips]
            })
            
            if logger:
                img_timestamp = get_timestamp(img_name)
                processing_time = time.time() - img_start_time
                num_associations = len([t for t in curr_tips if t.get("assoc_cost", -1.0) != -1.0])
                logger.log_image_processing(
                    image_name=img_name,
                    num_tips=len(curr_tips),
                    processing_time=processing_time,
                    image_timestamp=img_timestamp,
                    num_associations=num_associations,
                    additional_metrics={"plant_id": pid}
                )
            
            prev_tips = curr_tips

    # Save output
    with open(output_json, 'w') as f:
        json.dump(all_tracks, f, indent=2)
    print(f"Saved {len(all_tracks)} frames of tracks to {output_json}")

    if output_links_json:
        with open(output_links_json, 'w') as f:
            json.dump(all_positive_links, f, indent=2)
        print(f"Saved {len(all_positive_links)} positive associations to {output_links_json}")
    
    if logger:
        logger.save()
