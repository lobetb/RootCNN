
import json
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

def load_tracks(tracks_path):
    with open(tracks_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} frames from {tracks_path}")
    return data

def get_root_bases(tracks_data):
    bases = {}
    for frame in tracks_data:
        frame_tips = frame.get('tips', [])
        for tip in frame_tips:
            rid = tip['id']
            if rid not in bases:
                bases[rid] = (int(tip['x']), int(tip['y']))
    return bases

class RootImageDataset(Dataset):
    def __init__(self, tracks_data, img_folder, bases, downscale=0.25):
        self.tracks = tracks_data
        self.img_folder = Path(img_folder)
        self.bases = bases
        self.downscale = downscale
        
    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        frame = self.tracks[idx]
        basename = frame['basename']
        
        # We need to find the image
        # This might be slow if we glob every time. 
        # Ideally we pre-index, but for simplicity let's assume flat or known structure
        img_path = self.img_folder / basename
        if not img_path.exists():
            # Fallback scan (only do this once in reality, but for now...)
            # We'll just return None-like if missing, but better to crash or handle
            # Let's hope paths are correct.
            pass
            
        # Load Image
        try:
            pil_img = Image.open(img_path).convert('L')
            img_arr = np.array(pil_img)
        except:
            # Return dummy
            return torch.zeros(1, 1, 1), [], [], basename, 0.0

        H, W = img_arr.shape
        
        # Resize
        if self.downscale != 1.0:
            new_size = (int(W * self.downscale), int(H * self.downscale))
            img_arr = cv2.resize(img_arr, new_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img_arr).float() / 255.0
        
        # Invert for cost (Roots are dark=0 -> Cost 0? No, Distance Transform cost)
        # We want to minimize distance.
        # Cost to travel through pixel.
        # Root (dark) -> Low Cost.
        # Background (bright) -> High Cost.
        # img_tensor is 0(dark)..1(bright).
        # So Cost = img_tensor.
        # Add epsilon
        cost_tensor = img_tensor + 1e-4
        
        # Prepare Roots
        tips = frame.get('tips', [])
        
        # We need to return coords scaled
        root_ids = []
        bases_scaled = []
        tips_scaled = []
        
        for t in tips:
            rid = t['id']
            if rid not in self.bases: continue
            
            root_ids.append(rid)
            
            gx, gy = t['x'], t['y']
            bx, by = self.bases[rid]
            
            tips_scaled.append([gx * self.downscale, gy * self.downscale])
            bases_scaled.append([bx * self.downscale, by * self.downscale])
            
        return cost_tensor.unsqueeze(0), torch.tensor(root_ids), torch.tensor(bases_scaled), torch.tensor(tips_scaled), basename, self.downscale

def collate_fn(batch):
    # We only support batch size 1 for now because roots count varies
    return batch[0]


class MultiscaleGeodesicSolver(nn.Module):
    def __init__(self, scales=[0.125, 0.25, 0.5, 1.0], base_iter=1000, refine_iter=100):
        super().__init__()
        self.scales = scales
        self.base_iter = base_iter
        self.refine_iter = refine_iter
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
    def solve_at_scale(self, dist, cost, iters):
        # Iterative update: D = min(D, MinPool(D) + Cost)
        # MinPool(D) = -MaxPool(-D)
        for _ in range(iters):
            d_neighbor = -self.pool(-dist)
            d_prop = d_neighbor + cost
            dist = torch.min(dist, d_prop)
        return dist

    def forward(self, cost_map_full, source_coords, max_dist=10000):
        """
        Multiscale solver.
        cost_map_full: (1, H, W)
        source_coords: (N, 2)
        """
        device = cost_map_full.device
        N = source_coords.shape[0]
    def forward(self, cost_map_full, source_coords, max_dist=10000):
        """
        Multiscale solver.
        cost_map_full: (1, H, W) (Singleton channel, not expanded)
        source_coords: (N, 2)
        """
        device = cost_map_full.device
        N = source_coords.shape[0]
        # Ensure input is (1, 1, H, W)
        if cost_map_full.dim() == 3:
            cost_full = cost_map_full.unsqueeze(0)
        else:
            cost_full = cost_map_full
            
        _, _, H_full, W_full = cost_full.shape
        
        # 2. Initialize distance map at lowest scale
        current_dist = None
        
        for scale in self.scales:
            # Scale dimensions
            H_s = int(H_full * scale)
            W_s = int(W_full * scale)
            if H_s < 4 or W_s < 4: continue
            
            # Downscale Cost Map (Singleton)
            cost_s = F.interpolate(cost_full, size=(H_s, W_s), mode='bilinear', align_corners=False)
            
            # Initialize Dist
            if current_dist is None:
                # First scale: Init with infinity, shape (N, 1, H_s, W_s)
                current_dist = torch.full((N, 1, H_s, W_s), 1e5, device=device)
                
                # Set Sources
                sx = (source_coords[:, 0] * scale).long().clamp(0, W_s-1)
                sy = (source_coords[:, 1] * scale).long().clamp(0, H_s-1)
                current_dist[torch.arange(N, device=device), 0, sy, sx] = 0.0
                
                # Run base iterations
                iters = self.base_iter
            else:
                # Upsample previous result
                current_dist = F.interpolate(current_dist, size=(H_s, W_s), mode='bilinear', align_corners=False)
                
                # Rescale values
                ratio = scale / (self.scales[self.scales.index(scale)-1] if self.scales.index(scale)>0 else scale)
                current_dist = current_dist * ratio
                
                # Re-enforce sources
                sx = (source_coords[:, 0] * scale).long().clamp(0, W_s-1)
                sy = (source_coords[:, 1] * scale).long().clamp(0, H_s-1)
                current_dist[torch.arange(N, device=device), 0, sy, sx] = 0.0
                
                iters = self.refine_iter

            # Run Solver (Broadcasting cost_s)
            current_dist = self.solve_at_scale(current_dist, cost_s, iters)
            
        # Remove channel dim for return: (N, H, W)
        return current_dist.squeeze(1)

def run_gpu_measurement(tracks_file, img_folder, output_file, downscale=0.25, iterations=2000, batch_roots=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    data = load_tracks(tracks_file)
    bases = get_root_bases(data)
    
    dataset = RootImageDataset(data, img_folder, bases, downscale)
    # Batch size 1 image
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Multiscale solver
    solver = MultiscaleGeodesicSolver(scales=[0.125, 0.25, 0.5, 1.0], base_iter=1000, refine_iter=100).to(device)
    
    results = []
    
    for cost_tensor, rids, bases_scaled, tips_scaled, basename, ds in tqdm(loader, desc="GPU Processing"):
        if cost_tensor.shape[-1] < 10: 
            continue
        if len(rids) == 0:
            continue
            
        cost_tensor = cost_tensor.to(device)
        if cost_tensor.dim() == 3:
            cost_tensor = cost_tensor.unsqueeze(0) # (1, 1, H, W)
            
        _, _, H, W = cost_tensor.shape
        
        # 1. Sort roots by X coordinate to group spatially
        # tips_scaled: (N, 2) [x, y]
        # We sort by Tip X (or Base X? Tip X makes sense for current activity)
        # Let's sort by Average X of base and tip? Or just Tip X.
        sort_idxs = torch.argsort(tips_scaled[:, 0])
        
        rids = rids[sort_idxs]
        bases_scaled = bases_scaled[sort_idxs].to(device)
        tips_scaled = tips_scaled[sort_idxs].to(device)
        
        num_roots = len(rids)
        frame_measurements = []
        
        for i in range(0, num_roots, batch_roots):
            chunk_rids = rids[i : i + batch_roots]
            chunk_bases = bases_scaled[i : i + batch_roots]
            chunk_tips = tips_scaled[i : i + batch_roots]
            
            # 2. Compute ROI for this chunk
            # Min/Max of Bases AND Tips
            all_pts = torch.cat([chunk_bases, chunk_tips], dim=0)
            min_x = int(all_pts[:, 0].min().item())
            max_x = int(all_pts[:, 0].max().item())
            min_y = int(all_pts[:, 1].min().item())
            max_y = int(all_pts[:, 1].max().item())
            
            # Add padding (e.g. 50 pixels at this scale)
            # 50px at 0.25 scale = 200px real. Should be enough for path wiggle.
            padding = 50
            x1 = max(0, min_x - padding)
            y1 = max(0, min_y - padding)
            x2 = min(W, max_x + padding)
            y2 = min(H, max_y + padding)
            
            # Crop Cost Map
            # cost_tensor: (1, 1, H, W)
            cost_chunk = cost_tensor[:, :, y1:y2, x1:x2]
            
            # Adjust Coordinates to Crop
            # We need to subtract (x1, y1)
            chunk_bases_crop = chunk_bases.clone()
            chunk_bases_crop[:, 0] -= x1
            chunk_bases_crop[:, 1] -= y1
            
            chunk_tips_crop = chunk_tips.clone()
            chunk_tips_crop[:, 0] -= x1
            chunk_tips_crop[:, 1] -= y1
            
            with torch.no_grad():
                # Run Solver on CROP
                d_fields = solver(cost_chunk, chunk_bases_crop)
                
                # Read values at tips (Relative coords)
                tx = chunk_tips_crop[:, 0].long().clamp(0, cost_chunk.shape[-1]-1)
                ty = chunk_tips_crop[:, 1].long().clamp(0, cost_chunk.shape[-2]-1)
                
                lengths = d_fields[torch.arange(len(chunk_rids), device=device), ty, tx]
                
            lengths_cpu = lengths.cpu().numpy()
            rids_cpu = chunk_rids.numpy()
            
            for j, rid in enumerate(rids_cpu):
                l_px = lengths_cpu[j]
                l_global = l_px / ds
                if l_px > 90000: l_global = 0.0
                
                frame_measurements.append({
                    "root_id": int(rid),
                    "length_px": float(l_global)
                })

            del d_fields
            del cost_chunk
            torch.cuda.empty_cache()
            
        results.append({"image": basename, "measurements": frame_measurements})
        
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved GPU measurements to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--downscale", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=3000)
    
    args = parser.parse_args()
    run_gpu_measurement(args.tracks, args.images, args.output, args.downscale, args.iterations)
