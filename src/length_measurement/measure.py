
import json
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from skimage.filters import frangi
from skimage.graph import MCP_Geometric
from PIL import Image

def load_tracks(tracks_path):
    with open(tracks_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} frames from {tracks_path}")
    return data

def get_root_bases(tracks_data):
    """
    Identify the starting position (Base) for each root ID.
    The base is defined as the (x, y) position of the tip in the first frame it appears.
    Returns: dict {root_id: (x, y)}
    """
    bases = {}
    # Process frames in chronological order to find the first appearance
    # tracks_data is assumed to be a list of frames, which might not be sorted by time
    # provided it's the output of association.py, let's assume it is or sort it by filename/timestamp if needed.
    # But usually tracks.json is sorted.
    
    for frame in tracks_data:
        frame_tips = frame.get('tips', [])
        for tip in frame_tips:
            rid = tip['id']
            if rid not in bases:
                bases[rid] = (int(tip['x']), int(tip['y']))
    
    print(f"Identified {len(bases)} unique root IDs.")
    return bases


def compute_cost_map(image_arr, prev_paths=None, active_ids=None, sigmas=range(1, 4), alpha=0.01, use_frangi=True):
    """
    Compute the cost map for pathfinding.
    Combined cost = (1 - Vesselness) + alpha * (Distance to previous paths)
    """
    if use_frangi:
        # 1. Vesselness Cost
        # Frangi returns [0, 1] response
        vesselness = frangi(image_arr, sigmas=sigmas, black_ridges=True)
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-6)
        
        # Base cost: Background=1, Root=0
        base_cost = 1.0 - vesselness + 1e-4
    else:
        # Fast mode: Use raw inverted intensity
        if image_arr.max() > 1.0:
            norm_img = image_arr / 255.0
        else:
            norm_img = image_arr
            
        base_cost = norm_img + 1e-4
        vesselness = 1.0 - norm_img # Pseudo-vesselness for viz

    # 2. Temporal Cost (if applicable)
    if prev_paths and active_ids:
        H, W = image_arr.shape
        # Create a mask of all relevant previous paths
        mask = np.zeros((H, W), dtype=np.uint8)
        has_history = False
        
        for rid in active_ids:
            if rid in prev_paths:
                pts = np.array(prev_paths[rid], dtype=np.int32).reshape((-1, 1, 2))
                # Draw lines for the previous path
                cv2.polylines(mask, [pts], False, 255, 1) # 1px thickness is enough for distance transform
                has_history = True
        
        if has_history:
            # Distance transform: distance to nearest non-zero pixel
            # We want distance to the WHITE lines (255).
            # cv2.distanceTransform calculates distance to nearest ZERO pixel.
            # So we invert the mask: 0 (lines) -> 0 distance.
            inv_mask = 255 - mask
            dist_map = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
            
            # Normalize dist map or just use raw pixels?
            # Raw pixels is fine: being 10 pixels away adds 10*alpha to cost.
            # If alpha=0.01, 10 pixels = 0.1 cost penalty.
            # Vesselness difference is usually 0.5-0.8.
            # So 50-80 pixels away is enough to outweigh being on a root?
            # Maybe alpha should be higher, like 0.1?
            # Let's use user definable alpha.
            
            total_cost = base_cost + (dist_map * alpha)
            return total_cost, vesselness
            
    return base_cost, vesselness

def compute_geodesic_length(cost_map, start, end):
    """
    Compute the shortest path length from start to end on the cost map.
    start: (x, y)
    end: (x, y)
    """
    # MCP expects coordinates as (row, col) i.e. (y, x)
    start_node = (start[1], start[0])
    end_node = (end[1], end[0])
    
    # Check bounds
    H, W = cost_map.shape
    if not (0 <= start_node[0] < H and 0 <= start_node[1] < W):
        return None, []
    if not (0 <= end_node[0] < H and 0 <= end_node[1] < W):
        return None, []

    mcp = MCP_Geometric(cost_map)
    cumulative_costs, traceback_map = mcp.find_costs(starts=[start_node], ends=[end_node])
    
    # traceback returns a list of (row, col) tuples
    try:
        path = mcp.traceback(end_node)
    except ValueError:
        # Path not found
        return None, []
        
    # Compute Euclidean arc length of the path
    # path is [(y1, x1), (y2, x2), ...]
    path_arr = np.array(path)
    if len(path_arr) < 2:
        return 0.0, path
        
    diffs = path_arr[:-1] - path_arr[1:]
    dists = np.sqrt((diffs**2).sum(axis=1))
    total_length = dists.sum()
    
    # Convert path back to (x, y) for visualization/output
    path_xy = [(int(p[1]), int(p[0])) for p in path]
    
    return total_length, path_xy

def get_root_bases(tracks_data):
    """
    Identify the starting position (Base) for each root ID.
    But more importantly, track the EVOLUTION of the base.
    Actually, usually the base stays put.
    But for robustness, we can update the base if the root grows?
    No, base is base.
    Returns: dict {root_id: (x, y)}
    """
    bases = {}
    for frame in tracks_data:
        frame_tips = frame.get('tips', [])
        for tip in frame_tips:
            rid = tip['id']
            if rid not in bases:
                bases[rid] = (int(tip['x']), int(tip['y']))
    print(f"Identified {len(bases)} unique root IDs.")
    return bases

def process_series(tracks_file, img_folder, output_file, sigmas=range(1, 4), alpha=0.05, save_debug=False, downscale=0.25, use_frangi=True):
    tracks_data = load_tracks(tracks_file)
    bases = get_root_bases(tracks_data)
    
    img_folder = Path(img_folder)
    output_data = [] 
    
    prev_paths = {}
    
    if save_debug:
        debug_dir = Path("debug_length")
        debug_dir.mkdir(exist_ok=True)
        
    for frame_idx, frame in enumerate(tqdm(tracks_data, desc="Processing Frames")):
        basename = frame['basename']
        img_path = img_folder / basename
        
        if not img_path.exists():
            found = list(img_folder.rglob(basename))
            if found:
                img_path = found[0]
            else:
                continue
                
        pil_img = Image.open(img_path).convert('L')
        img_arr = np.array(pil_img) # (H, W)
        H_full, W_full = img_arr.shape
        
        # 1. Determine ROI (Bounding Box of all tips + Base)
        # We need to include bases and tips of CURRENT frame
        frame_tips = frame.get('tips', [])
        if not frame_tips:
            continue
            
        all_xs = []
        all_ys = []
        frame_ids = []
        
        for t in frame_tips:
            rid = t['id']
            frame_ids.append(rid)
            all_xs.append(t['x'])
            all_ys.append(t['y'])
            if rid in bases:
                all_xs.append(bases[rid][0])
                all_ys.append(bases[rid][1])
        
        if not all_xs:
            continue
            
        min_x, max_x = min(all_xs), max(all_xs)
        min_y, max_y = min(all_ys), max(all_ys)
        
        # Add padding
        padding = 200
        x1 = max(0, int(min_x - padding))
        y1 = max(0, int(min_y - padding))
        x2 = min(W_full, int(max_x + padding))
        y2 = min(H_full, int(max_y + padding))
        
        # Crop
        crop_arr = img_arr[y1:y2, x1:x2]
        crop_h, crop_w = crop_arr.shape
        
        # 2. Downscale Crop
        if downscale != 1.0:
            new_size = (int(crop_w * downscale), int(crop_h * downscale))
            crop_resized = cv2.resize(crop_arr, new_size, interpolation=cv2.INTER_AREA)
        else:
            crop_resized = crop_arr
            
        # Adjust prev_paths for crop & scale
        # We need to map global previous paths to local resized coordinates
        # prev_paths stores GLOBAL (x, y)
        prev_paths_local = {}
        for rid, path in prev_paths.items():
            if rid not in frame_ids: continue
            # Transform path
            local_path = []
            for (px, py) in path:
                lx = (px - x1) * downscale
                ly = (py - y1) * downscale
                local_path.append((lx, ly))
            prev_paths_local[rid] = local_path
            
        # Compute Cost Map on smaller image
        cost_map, v_map = compute_cost_map(crop_resized, prev_paths_local, frame_ids, sigmas=sigmas, alpha=alpha, use_frangi=use_frangi)
        
        frame_results = {
            "image": str(basename),
            "measurements": []
        }
        
        debug_img_color = None
        if save_debug:
            debug_img_color = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)
            # Normalize cost map for viz
            v_min, v_max = cost_map.min(), cost_map.max()
            if v_max - v_min > 1e-6:
                cm_norm = (cost_map - v_min) / (v_max - v_min)
            else:
                cm_norm = np.zeros_like(cost_map)
                
            cm_color = cv2.applyColorMap((cm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            debug_img_color = cv2.addWeighted(debug_img_color, 0.7, cm_color, 0.3, 0)
        
        current_frame_paths = {}
        
        for tip in frame_tips:
            rid = tip['id']
            # Global Coords
            g_tip_x, g_tip_y = tip['x'], tip['y']
            g_base_x, g_base_y = bases[rid]
            
            # Local Resized Coords
            l_tip_x = (g_tip_x - x1) * downscale
            l_tip_y = (g_tip_y - y1) * downscale
            l_base_x = (g_base_x - x1) * downscale
            l_base_y = (g_base_y - y1) * downscale
            
            # Compute path
            length_local, path_local = compute_geodesic_length(cost_map, (l_base_x, l_base_y), (l_tip_x, l_tip_y))
            
            if length_local is not None:
                # Convert length back to global scale
                length_global = length_local / downscale
                
                # Convert path back to global coords
                path_global = []
                for (lx, ly) in path_local:
                    gx = (lx / downscale) + x1
                    gy = (ly / downscale) + y1
                    path_global.append((gx, gy))
                
                frame_results["measurements"].append({
                    "root_id": rid,
                    "tip_x": g_tip_x,
                    "tip_y": g_tip_y,
                    "base_x": g_base_x,
                    "base_y": g_base_y,
                    "length_px": float(length_global)
                })
                
                current_frame_paths[rid] = path_global
                
                if save_debug and debug_img_color is not None:
                    path_pts = np.array(path_local, dtype=np.int32).reshape((-1, 1, 2))
                    color = (0, 0, 255) if rid % 2 == 0 else (0, 255, 255)
                    cv2.polylines(debug_img_color, [path_pts], False, color, 1) # thiner line for smaller img
                    cv2.putText(debug_img_color, f"ID{rid}", (int(l_tip_x), int(l_tip_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        prev_paths.update(current_frame_paths)
        output_data.append(frame_results)
        
        if save_debug:
            cv2.imwrite(str(debug_dir / f"debug_{frame_idx:04d}_{basename}"), debug_img_color)
            
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved measurements to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure root lengths using Geodesic Path on Vesselness map.")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json")
    parser.add_argument("--images", required=True, help="Path to folder containing images")
    parser.add_argument("--output", required=True, help="Path to output json file")
    parser.add_argument("--sigma-min", type=int, default=1, help="Min sigma for Frangi")
    parser.add_argument("--sigma-max", type=int, default=4, help="Max sigma for Frangi")
    parser.add_argument("--debug", action="store_true", help="Save debug images with drawn paths")
    parser.add_argument("--downscale", type=float, default=0.25, help="Downscale factor (default 0.25)")
    parser.add_argument("--fast", action="store_true", help="Use raw intensity instead of Frangi filter (faster)")
    
    args = parser.parse_args()
    
    sigmas = range(args.sigma_min, args.sigma_max)
    process_series(args.tracks, args.images, args.output, sigmas=sigmas, save_debug=args.debug, downscale=args.downscale, use_frangi=not args.fast)
