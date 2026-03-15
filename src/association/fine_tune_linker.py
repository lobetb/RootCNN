import torch
import torch.nn as nn
import json
import os
import sys
import numpy as np

# Add the project root to the python path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import DataLoader
from src.association.models import AffinityMLP, LinkingDataset
from src.association.gnn_models import TipGNN
from src.association.gnn_utils import build_graph_batch, resolve_link_indices
from src.utils.common import get_device, get_time_diff_hours

def fine_tune_model(model_path, pos_links_json, neg_links_json, output_path, epochs=10, lr=1e-4, neg_weight=5.0):
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine model type from checkpoint keys
    keys = list(checkpoint.keys())
    is_mlp = any(k.startswith('net.') for k in keys)
    
    if is_mlp:
        print("Detected AffinityMLP model.")
        model = AffinityMLP().to(device)
    else:
        print("Detected TipGNN model.")
        model = TipGNN().to(device)
        
    model.load_state_dict(checkpoint)
    
    # If it's an MLP, remove the Sigmoid [index 8 or 9] for training stability
    # AffinityMLP has Sigmoid at the end of nn.Sequential
    if is_mlp:
        print("Removing Sigmoid for training stability...")
        layers = list(model.net.children())
        if isinstance(layers[-1], nn.Sigmoid):
            model.net = nn.Sequential(*layers[:-1])
    
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Load data
    with open(pos_links_json, 'r') as f:
        pos_data = json.load(f)
        
    # Handle flat list output from tracker for positives
    if isinstance(pos_data, list):
        new_pos_data = {}
        for item in pos_data:
            if 'tip1' in item and 'tip2' in item:
                img1 = os.path.basename(item['tip1']['image'])
                img2 = os.path.basename(item['tip2']['image'])
                pk = f"{img1}->{img2}"
                if pk not in new_pos_data: new_pos_data[pk] = []
                
                new_pos_data[pk].append({
                    'tip1_x': item['tip1']['x'],
                    'tip1_y': item['tip1']['y'],
                    'tip2_x': item['tip2']['x'],
                    'tip2_y': item['tip2']['y'],
                    'tip1_features': item['tip1'].get('features', []),
                    'tip2_features': item['tip2'].get('features', []),
                })
        pos_data = new_pos_data
        
    with open(neg_links_json, 'r') as f:
        neg_data = json.load(f)
        
    # We need features for the tips. 
    # Assuming consolidated links format (which includes features)
    
    all_pairs = []
    
    # Process positives
    for pk_raw, links in pos_data.items():
        if "->" not in pk_raw: continue
        # Extract features and coordinates from links
        tips1 = []
        tips2 = []
        true_links = []
        
        # Canonicalize pk
        i1, i2 = pk_raw.split("->")
        pk = f"{os.path.basename(i1)}->{os.path.basename(i2)}"
        
        # We need to reconstruct the tips list for build_graph_batch
        # This is a bit complex because one link entry refers to one tip, 
        # but build_graph_batch expects a list of ALL tips in the image.
        # However, for fine-tuning on SPECIFIC links, we can just use the provided tips.
        
        seen1, seen2 = {}, {}
        for l in links:
            if l.get('flagged_negative', False):
                continue
                
            t1 = {'x': l['tip1_x'], 'y': l['tip1_y'], 'features': l.get('tip1_features', [])}
            t2 = {'x': l['tip2_x'], 'y': l['tip2_y'], 'features': l.get('tip2_features', [])}
            
            k1, k2 = (t1['x'], t1['y']), (t2['x'], t2['y'])
            if k1 not in seen1: seen1[k1] = len(tips1); tips1.append(t1)
            if k2 not in seen2: seen2[k2] = len(tips2); tips2.append(t2)
            
            true_links.append((seen1[k1], seen2[k2]))
            
        all_pairs.append({'tips1': tips1, 'tips2': tips2, 'links': true_links, 'pk': pk, 'is_neg': False})

    # Process explicit negatives
    for pk_raw, links in neg_data.items():
        if "->" not in pk_raw: continue
        i1, i2 = pk_raw.split("->")
        pk = f"{os.path.basename(i1)}->{os.path.basename(i2)}"
        
        tips1, tips2 = [], []
        false_links = []
        seen1, seen2 = {}, {}
        for l in links:
            t1 = {'x': l['tip1_x'], 'y': l['tip1_y'], 'features': l.get('tip1_features', [])}
            t2 = {'x': l['tip2_x'], 'y': l['tip2_y'], 'features': l.get('tip2_features', [])}
            k1, k2 = (t1['x'], t1['y']), (t2['x'], t2['y'])
            if k1 not in seen1: seen1[k1] = len(tips1); tips1.append(t1)
            if k2 not in seen2: seen2[k2] = len(tips2); tips2.append(t2)
            false_links.append((seen1[k1], seen2[k2]))
            
        all_pairs.append({'tips1': tips1, 'tips2': tips2, 'links': false_links, 'pk': pk, 'is_neg': True})
    p_count = len([p for p in all_pairs if not p['is_neg']])
    n_count = len([p for p in all_pairs if p['is_neg']])
    print(f"Data Distribution: {p_count} positive pairs, {n_count} negative pairs")

    for epoch in range(epochs):
        epoch_loss = 0
        np.random.shuffle(all_pairs)
        
        for pair in all_pairs:
            img1, img2 = pair['pk'].split("->")
            t_delta = max(0.01, get_time_diff_hours(img1, img2))
            
            if is_mlp:
                # Prepare inputs for AffinityMLP
                if not pair['links']: continue
                
                batch_x = []
                batch_y = []
                
                for idx1, idx2 in pair['links']:
                    if idx1 >= len(pair['tips1']) or idx2 >= len(pair['tips2']): continue
                    t1 = pair['tips1'][idx1]
                    t2 = pair['tips2'][idx2]
                    
                    f1 = np.array(t1['features'])
                    f2 = np.array(t2['features'])
                    
                    if len(f1) == 0 or len(f2) == 0:
                        continue
                    
                    dx = t2['x'] - t1['x']
                    dy = t2['y'] - t1['y']
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    spatial = np.array([dx/t_delta/100.0, dy/t_delta/100.0, dist/t_delta/100.0])
                    x_feat = np.concatenate([f1, f2, spatial], axis=0)
                    batch_x.append(x_feat)
                    batch_y.append(0.0 if pair['is_neg'] else 1.0)
                    
                if not batch_x: continue
                
                x_tensor = torch.tensor(np.array(batch_x), device=device).float()
                y_tensor = torch.tensor(batch_y, device=device).float().unsqueeze(1)
                
                weight = neg_weight if pair['is_neg'] else 1.0
                
                optimizer.zero_grad()
                preds = model(x_tensor)
                
                # Use BCEWithLogitsLoss since we removed Sigmoid
                loss = nn.functional.binary_cross_entropy_with_logits(preds, y_tensor, reduction='mean') * weight
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            else:
                # Prepare inputs for TipGNN
                batch = build_graph_batch(pair['tips1'], pair['tips2'], pair['links'], device, time_diff=t_delta)
                if batch is None: continue
                
                node_feats, edge_index, edge_spatials, labels = batch
                
                # If this was a negative pair, flip labels to 0.0
                if pair['is_neg']:
                    labels = torch.zeros_like(labels)
                    weight = neg_weight
                else:
                    weight = 1.0
                    
                optimizer.zero_grad()
                preds = model(node_feats, edge_index, edge_spatials)
                
                # Binary Cross Entropy with logits for GNN
                loss = nn.functional.binary_cross_entropy_with_logits(preds, labels, reduction='mean') * weight
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/max(1, len(all_pairs)):.6f}")

    torch.save(model.state_dict(), output_path)
    print(f"Fine-tuned model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--positives", required=True)
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--output", default="models/fine_tuned_linker.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    fine_tune_model(args.model, args.positives, args.negatives, args.output, epochs=args.epochs, lr=args.lr)
