import json
import csv
import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

# Add the project root to the python path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.length_measurement.measure_utils as utils

PX_PER_CM = 314
PX_PER_MM = PX_PER_CM / 10.0

def compute_incremental_speeds(tracks_file, img_folder, output_csv, downscale=0.25, use_frangi=True, exponent=4, sigmas=None, use_threshold=False, stop_event=None, **kwargs):
    print(f"Loading tracks from {tracks_file}...")
    with open(tracks_file, 'r') as f:
        tracks_data = json.load(f)
        
    img_folder = Path(img_folder)
    if not img_folder.exists():
        print(f"Error: Images folder does not exist: {img_folder}")
        return

    # Organize by Root ID
    # roots[root_id] = [(timestamp, x, y, basename), ...]
    roots = defaultdict(list)
    
    ts_fail_count = 0
    for frame in tracks_data:
        basename = frame['basename']
        ts = utils.parse_timestamp(basename)
        if ts is None:
            ts_fail_count += 1
            continue
            
        # Extract plant ID: first 6 characters before the first "-"
        plant_id = basename.split('-')[0][:6]

        for tip in frame.get('tips', []):
            rid = f"{plant_id}_{tip['id']}"
            roots[rid].append({
                'ts': ts,
                'x': tip['x'],
                'y': tip['y'],
                'basename': basename,
                'plant_id': plant_id
            })
    
    if ts_fail_count > 0:
        print(f"Warning: Could not parse timestamp for {ts_fail_count} frames.")
            
    print(f"Processing {len(roots)} roots...")
    
    # Group tips by basename
    # img_to_tips[basename] = {rid: (x, y)}
    img_to_tips = defaultdict(dict)
    for rid, history in roots.items():
        for entry in history:
            img_to_tips[entry['basename']][rid] = (entry['x'], entry['y'])
    
    unique_basenames = sorted(img_to_tips.keys(), key=lambda b: utils.parse_timestamp(b))
    
    # Store total lengths: length_cache[rid][basename] = total_geodesic_length
    length_cache = defaultdict(dict)
    missing_files = set()
    
    print(f"Computing geodesic lengths from plug for {len(unique_basenames)} images...")
    
    for basename in tqdm(unique_basenames, desc="Processing Images"):
        if stop_event and stop_event.is_set():
            break
            
        img_path = img_folder / basename
        if not img_path.exists():
            missing_files.add(str(img_path))
            continue
            
        # Load and process image
        pil_img = Image.open(img_path).convert('L')
        img_arr = np.array(pil_img)
        
        # Detect Plug
        y_bound = utils.find_support_boundary(img_arr)
        scan_y, plug_x = utils.detect_plug(img_arr, y_bound)
        
        if scan_y is None:
            # We skip this image if plug is not found
            continue
            
        # Downscale for cost map
        if downscale != 1.0:
            new_size = (int(img_arr.shape[1] * downscale), int(img_arr.shape[0] * downscale))
            img_resized = cv2.resize(img_arr, new_size, interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_arr
            
        # Compute Cost Map for the whole image
        cost_map, _ = utils.compute_cost_map(img_resized, use_frangi=use_frangi, exponent=exponent, sigmas=sigmas, use_threshold=use_threshold)
        
        # Geodesic search from Plug
        plug_node = (int(scan_y * downscale), int(plug_x * downscale))
        mcp = utils.MCP_Geometric(cost_map)
        try:
            # Pre-calculate costs from plug to everywhere
            mcp.find_costs(starts=[plug_node])
        except (ValueError, IndexError):
            continue
            
        # Extract distance for each tip in this image
        for rid, (tx, ty) in img_to_tips[basename].items():
            tip_node = (int(ty * downscale), int(tx * downscale))
            try:
                # We want Euclidean arc length, so we must traceback
                path = mcp.traceback(tip_node)
                path_arr = np.array(path)
                if len(path_arr) < 2:
                    dist = 0.0
                else:
                    diffs = path_arr[:-1] - path_arr[1:]
                    dists = np.sqrt((diffs**2).sum(axis=1))
                    dist = dists.sum() / downscale
                length_cache[rid][basename] = dist
            except (ValueError, IndexError):
                # Path not found to this tip
                continue

    # Now compute incremental speeds based on total length changes
    rows = []
    for rid, history in roots.items():
        history.sort(key=lambda x: x['ts'])
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            p_len = length_cache[rid].get(prev['basename'])
            c_len = length_cache[rid].get(curr['basename'])
            
            if p_len is None or c_len is None:
                continue
                
            dt_seconds = (curr['ts'] - prev['ts']).total_seconds()
            if dt_seconds <= 0:
                continue
            dt_hours = dt_seconds / 3600.0
            
            growth_px = c_len - p_len
            # In some cases, growth could be negative due to noise or tip error, 
            # we keep it for now but user might want to filter small negatives
            speed_px = growth_px / dt_hours
            speed_mm = speed_px / PX_PER_MM
            
            rows.append({
                "plant_id": curr['plant_id'],
                "root_id": rid,
                "image_prev": prev['basename'],
                "image_curr": curr['basename'],
                "time_prev": prev['ts'].isoformat(),
                "time_curr": curr['ts'].isoformat(),
                "dt_hours": round(dt_hours, 4),
                "growth_px": round(growth_px, 2),
                "speed_px_per_hour": round(speed_px, 2),
                "speed_mm_per_hour": round(speed_mm, 4)
            })

    # Save CSV
    if missing_files:
        print(f"Warning: {len(missing_files)} images were missing.")
        
    if not rows:
        print("No speeds computed. Reasons could be:")
        print("- All roots have only 1 frame of history")
        print("- Images were not found in the provided folder")
        print("- Plug detection failed for some images")
        print("- Geodesic paths from plug to tips were not found")
        return
        
    headers = [
        "plant_id", "root_id", "time_curr", "speed_mm_per_hour", "speed_px_per_hour",
        "growth_px", "dt_hours", 
        "time_prev", "image_prev", "image_curr"
    ]
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Saved {len(rows)} speed measurements to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute root growth speeds using incremental geodesic distance.")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json")
    parser.add_argument("--images", required=True, help="Path to images folder")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--downscale", type=float, default=0.25, help="Downscale factor")
    parser.add_argument("--fast", action="store_true", help="Use raw intensity instead of Frangi")
    parser.add_argument("--exponent", type=float, default=4.0, help="Exponent for cost non-linearity")
    parser.add_argument("--binary", action="store_true", help="Apply strict binary thresholding before pathfinding")
    args = parser.parse_args()
    compute_incremental_speeds(args.tracks, args.images, args.output, downscale=args.downscale, use_frangi=not args.fast, exponent=args.exponent, use_threshold=args.binary)
