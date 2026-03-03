import torch
import numpy as np

def resolve_link_indices(links, tips1, tips2, tolerance=5):
    """
    Matches annotated links to detected tip features using coordinates if available.
    Falls back to indices if coordinates are missing or no match is found within tolerance.
    """
    resolved = set()
    
    # Pre-calculate coordinates for faster lookup
    coords1 = np.array([[t['x'], t['y']] for t in tips1]) if tips1 else np.array([]).reshape(0,2)
    coords2 = np.array([[t['x'], t['y']] for t in tips2]) if tips2 else np.array([]).reshape(0,2)
    
    for l in links:
        idx1 = l.get('tip1_index')
        idx2 = l.get('tip2_index')
        
        # Try coordinate matching if available
        if all(k in l for k in ['tip1_x', 'tip1_y', 'tip2_x', 'tip2_y']):
            x1, y1 = l['tip1_x'], l['tip1_y']
            x2, y2 = l['tip2_x'], l['tip2_y']
            
            # Find nearest feature in tips1
            if len(coords1) > 0:
                dists1 = np.sqrt((coords1[:, 0] - x1)**2 + (coords1[:, 1] - y1)**2)
                best1 = np.argmin(dists1)
                if dists1[best1] <= tolerance:
                    idx1 = int(best1)
                    
            # Find nearest feature in tips2
            if len(coords2) > 0:
                dists2 = np.sqrt((coords2[:, 0] - x2)**2 + (coords2[:, 1] - y2)**2)
                best2 = np.argmin(dists2)
                if dists2[best2] <= tolerance:
                    idx2 = int(best2)
        
        if idx1 is not None and idx2 is not None:
            resolved.add((idx1, idx2))
        
    return resolved

def build_graph_batch(tips1, tips2, true_links, device, spatial_threshold=150, time_diff=1.0):
    """
    Builds a graph for training the GNN linker.
    tips1, tips2: list of dicts with 'features', 'x', 'y'
    true_links: set of (idx1, idx2) representing ground truth matches
    
    Refinement: Only use tips that are part of at least one link in true_links.
    This avoids false negatives from unannotated tips.
    """
    if not true_links:
        return None

    # Safety Check: Filter true_links to only include valid indices
    # This prevents 'list index out of range' if the annotation file is out of sync
    valid_links = []
    skipped_count = 0
    for l in true_links:
        if l[0] < len(tips1) and l[1] < len(tips2):
            valid_links.append(l)
        else:
            skipped_count += 1
            
    if skipped_count > 0:
        # We can't easily use a logger here without passing it in, 
        # but we can at least print to console for visibility
        print(f"  [Warning] Skipped {skipped_count} links with out-of-bounds indices.")

    if not valid_links:
        return None

    # 1. Identify which tips from tips1 and tips2 are involved in valid links
    used_idx1 = sorted(list(set(l[0] for l in valid_links)))
    used_idx2 = sorted(list(set(l[1] for l in valid_links)))
    
    if not used_idx1 or not used_idx2:
        return None
        
    # Map old indices to new indices (0..N-1) for the filtered sets
    idx1_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_idx1)}
    idx2_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_idx2)}
    
    # Filter tips
    filtered_tips1 = [tips1[i] for i in used_idx1]
    filtered_tips2 = [tips2[i] for i in used_idx2]
    
    num1 = len(filtered_tips1)
    num2 = len(filtered_tips2)
    
    # 2. Prepare features and coordinates for filtered tips
    feats1 = torch.tensor([t['features'] for t in filtered_tips1], device=device).float()
    coords1 = torch.tensor([[t['x'], t['y']] for t in filtered_tips1], device=device).float()
    
    feats2 = torch.tensor([t['features'] for t in filtered_tips2], device=device).float()
    coords2 = torch.tensor([[t['x'], t['y']] for t in filtered_tips2], device=device).float()
    
    # 3. Build Edge Index based on spatial threshold
    dist_matrix = torch.cdist(coords1, coords2)
    mask = dist_matrix < spatial_threshold
    
    src_local, dst_local = torch.where(mask)
    if src_local.numel() == 0:
        return None
        
    # Global indices for the GNN: tips1 are 0..num1-1, tips2 are num1..num1+num2-1
    src_global = src_local
    dst_global = dst_local + num1
    
    edge_index = torch.stack([src_global, dst_global], dim=0)
    
    # 4. Edge spatial features
    # Normalize by time_diff (velocity)
    t_delta = max(0.01, time_diff)
    dxy = (coords2[dst_local] - coords1[src_local]) / t_delta
    dist = (dist_matrix[src_local, dst_local].unsqueeze(1)) / t_delta
    edge_spatials = torch.cat([dxy, dist], dim=1) / 100.0
    
    # 5. Node features
    node_feats = torch.cat([feats1, feats2], dim=0)
    
    # 6. Labels
    # For each edge in the graph, check if the original pair was a valid link
    labels = []
    # Map back to original indices to check against valid_links
    rev_idx1 = used_idx1
    rev_idx2 = used_idx2
    
    # Convert valid_links to set for fast lookup
    valid_links_set = set(valid_links)
    
    for s_new, d_new in zip(src_local.cpu().numpy(), dst_local.cpu().numpy()):
        s_old = rev_idx1[s_new]
        d_old = rev_idx2[d_new]
        if (s_old, d_old) in valid_links_set:
            labels.append(1.0)
        else:
            labels.append(0.0)
            
    labels = torch.tensor(labels, device=device).float().unsqueeze(1)
    
    return node_feats, edge_index, edge_spatials, labels
