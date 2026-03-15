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

def compute_incremental_speeds(tracks_file, img_folder, output_csv, downscale=0.25, use_frangi=True, stop_event=None, **kwargs):
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
    
    multi_frame_roots = [rid for rid, h in roots.items() if len(h) > 1]
    print(f"Found {len(multi_frame_roots)} roots with more than one frame of history.")
    
    rows = []
    missing_files = set()
    
    for rid, history in tqdm(roots.items(), desc="Computing Speeds"):
        if stop_event and stop_event.is_set():
            print("\n[STOP] Speed computation cancelled.")
            break

        # Sort by time
        history.sort(key=lambda x: x['ts'])
        
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            dt_seconds = (curr['ts'] - prev['ts']).total_seconds()
            if dt_seconds <= 0:
                continue
                
            dt_hours = dt_seconds / 3600.0
            
            # Compute geodesic distance from prev tip to curr tip in CURR image
            img_path = img_folder / curr['basename']
            if not img_path.exists():
                missing_files.add(str(img_path))
                continue
                
            pil_img = Image.open(img_path).convert('L')
            img_arr = np.array(pil_img)
            H_full, W_full = img_arr.shape
            
            # ROI for the two tips
            x1 = max(0, int(min(prev['x'], curr['x']) - 200))
            y1 = max(0, int(min(prev['y'], curr['y']) - 200))
            x2 = min(W_full, int(max(prev['x'], curr['x']) + 200))
            y2 = min(H_full, int(max(prev['y'], curr['y']) + 200))
            
            crop_arr = img_arr[y1:y2, x1:x2]
            
            if downscale != 1.0:
                new_size = (int(crop_arr.shape[1] * downscale), int(crop_arr.shape[0] * downscale))
                crop_resized = cv2.resize(crop_arr, new_size, interpolation=cv2.INTER_AREA)
            else:
                crop_resized = crop_arr
                
            # Local coords
            l_prev_x = (prev['x'] - x1) * downscale
            l_prev_y = (prev['y'] - y1) * downscale
            l_curr_x = (curr['x'] - x1) * downscale
            l_curr_y = (curr['y'] - y1) * downscale
            
            cost_map, _ = utils.compute_cost_map(crop_resized, use_frangi=use_frangi)
            length_local, _ = utils.compute_geodesic_length(cost_map, (l_prev_x, l_prev_y), (l_curr_x, l_curr_y))
            
            if length_local is not None:
                growth_px = length_local / downscale
                speed = growth_px / dt_hours
                
                rows.append({
                    "plant_id": curr['plant_id'],
                    "root_id": rid,
                    "image_prev": prev['basename'],
                    "image_curr": curr['basename'],
                    "time_prev": prev['ts'].isoformat(),
                    "time_curr": curr['ts'].isoformat(),
                    "dt_hours": round(dt_hours, 4),
                    "growth_px": round(growth_px, 2),
                    "speed_px_per_hour": round(speed, 2)
                })
                
    # Save CSV
    if missing_files:
        print(f"Warning: {len(missing_files)} images were missing. Example: {list(missing_files)[0]}")
        
    if not rows:
        print("No speeds computed. Reasons could be:")
        print("- All roots have only 1 frame of history")
        print("- Images were not found in the provided folder")
        print("- All growth deltas were invalid (geodesic path not found)")
        return
        
    headers = [
        "plant_id", "root_id", "time_curr", "speed_px_per_hour", 
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
    
    args = parser.parse_args()
    compute_incremental_speeds(args.tracks, args.images, args.output, downscale=args.downscale, use_frangi=not args.fast)
