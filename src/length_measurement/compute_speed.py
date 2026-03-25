import json
import csv
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _arc_length_from_path(path_rc, downscale):
    path_arr = np.array(path_rc)
    if len(path_arr) < 2:
        return 0.0
    diffs = path_arr[:-1] - path_arr[1:]
    dists = np.sqrt((diffs**2).sum(axis=1))
    return float(dists.sum() / downscale)


def _compute_lengths_in_region(
    img_arr,
    tips_xy,
    plug_xy,
    bounds,
    downscale,
    use_frangi,
    exponent,
    sigmas,
    use_threshold,
    use_ends_optimization,
):
    x1, y1, x2, y2 = bounds

    region_tips = {}
    for rid, (tx, ty) in tips_xy.items():
        if x1 <= tx < x2 and y1 <= ty < y2:
            region_tips[rid] = (tx, ty)

    if not region_tips:
        return {}

    crop = img_arr[y1:y2, x1:x2]
    if crop.size == 0:
        return {rid: None for rid in region_tips}

    if downscale != 1.0:
        new_size = (max(1, int(crop.shape[1] * downscale)), max(1, int(crop.shape[0] * downscale)))
        crop = cv2.resize(crop, new_size, interpolation=cv2.INTER_AREA)

    cost_map, _ = utils.compute_cost_map(
        crop,
        use_frangi=use_frangi,
        exponent=exponent,
        sigmas=sigmas,
        use_threshold=use_threshold,
    )

    H, W = cost_map.shape
    plug_x, plug_y = plug_xy
    plug_node = (int((plug_y - y1) * downscale), int((plug_x - x1) * downscale))
    plug_node = (min(max(plug_node[0], 0), H - 1), min(max(plug_node[1], 0), W - 1))

    tip_nodes = {}
    for rid, (tx, ty) in region_tips.items():
        node = (int((ty - y1) * downscale), int((tx - x1) * downscale))
        node = (min(max(node[0], 0), H - 1), min(max(node[1], 0), W - 1))
        tip_nodes[rid] = node

    mcp = utils.MCP_Geometric(cost_map)
    try:
        if use_ends_optimization:
            ends = list(dict.fromkeys(tip_nodes.values()))
            mcp.find_costs(starts=[plug_node], ends=ends)
        else:
            mcp.find_costs(starts=[plug_node])
    except TypeError:
        mcp.find_costs(starts=[plug_node])
    except (ValueError, IndexError):
        return {rid: None for rid in region_tips}

    out = {}
    for rid, node in tip_nodes.items():
        try:
            path = mcp.traceback(node)
            out[rid] = _arc_length_from_path(path, downscale)
        except (ValueError, IndexError):
            out[rid] = None

    return out


def _process_single_image_job(job):
    (
        basename,
        img_path_str,
        tips_xy,
        downscale,
        use_frangi,
        exponent,
        sigmas,
        use_threshold,
        optimize,
        roi_margin,
        fallback_full_frame,
        use_ends_optimization,
    ) = job

    t0 = time.time()
    img_path = Path(img_path_str)
    if not img_path.exists():
        return {
            "basename": basename,
            "dists": {},
            "missing_file": str(img_path),
            "plug_fail": False,
            "tips_count": len(tips_xy),
            "elapsed": max(time.time() - t0, 1e-6),
        }

    img_arr = np.array(Image.open(img_path).convert("L"))
    y_bound = utils.find_support_boundary(img_arr)
    scan_y, plug_x = utils.detect_plug(img_arr, y_bound)

    if scan_y is None:
        return {
            "basename": basename,
            "dists": {},
            "missing_file": None,
            "plug_fail": True,
            "tips_count": len(tips_xy),
            "elapsed": max(time.time() - t0, 1e-6),
        }

    if optimize:
        xs = [plug_x] + [p[0] for p in tips_xy.values()]
        ys = [scan_y] + [p[1] for p in tips_xy.values()]
        x1 = max(0, min(xs) - roi_margin)
        y1 = max(0, min(ys) - roi_margin)
        x2 = min(img_arr.shape[1], max(xs) + roi_margin)
        y2 = min(img_arr.shape[0], max(ys) + roi_margin)

        dists = _compute_lengths_in_region(
            img_arr,
            tips_xy,
            (plug_x, scan_y),
            (x1, y1, x2, y2),
            downscale,
            use_frangi,
            exponent,
            sigmas,
            use_threshold,
            use_ends_optimization,
        )

        if fallback_full_frame:
            missing = {rid: tips_xy[rid] for rid, dist in dists.items() if dist is None}
            if missing:
                full_dists = _compute_lengths_in_region(
                    img_arr,
                    missing,
                    (plug_x, scan_y),
                    (0, 0, img_arr.shape[1], img_arr.shape[0]),
                    downscale,
                    use_frangi,
                    exponent,
                    sigmas,
                    use_threshold,
                    use_ends_optimization,
                )
                for rid, dist in full_dists.items():
                    dists[rid] = dist
    else:
        dists = _compute_lengths_in_region(
            img_arr,
            tips_xy,
            (plug_x, scan_y),
            (0, 0, img_arr.shape[1], img_arr.shape[0]),
            downscale,
            use_frangi,
            exponent,
            sigmas,
            use_threshold,
            False,
        )

    return {
        "basename": basename,
        "dists": dists,
        "missing_file": None,
        "plug_fail": False,
        "tips_count": len(tips_xy),
        "elapsed": max(time.time() - t0, 1e-6),
    }

def compute_incremental_speeds(
    tracks_file,
    img_folder,
    output_csv,
    downscale=0.25,
    use_frangi=True,
    exponent=4,
    sigmas=None,
    use_threshold=False,
    optimize=True,
    roi_margin=220,
    fallback_full_frame=True,
    use_ends_optimization=True,
    workers=0,
    stop_event=None,
    **kwargs,
):
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
    plug_fail_count = 0

    if workers is None:
        workers = 0
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = int(workers)
    
    print(f"Computing geodesic lengths from plug for {len(unique_basenames)} images (workers={workers})...")

    jobs = []
    for basename in unique_basenames:
        img_path = img_folder / basename
        jobs.append(
            (
                basename,
                str(img_path),
                img_to_tips[basename],
                downscale,
                use_frangi,
                exponent,
                sigmas,
                use_threshold,
                optimize,
                roi_margin,
                fallback_full_frame,
                use_ends_optimization,
            )
        )

    if workers <= 1:
        pbar = tqdm(jobs, desc="Processing Images")
        for job in pbar:
            if stop_event and stop_event.is_set():
                break
            result = _process_single_image_job(job)

            if result["missing_file"]:
                missing_files.add(result["missing_file"])
                continue
            if result["plug_fail"]:
                plug_fail_count += 1
                continue

            basename = result["basename"]
            for rid, dist in result["dists"].items():
                if dist is not None:
                    length_cache[rid][basename] = dist

            elapsed = result["elapsed"]
            tips_count = result["tips_count"]
            pbar.set_postfix(frame_s=f"{elapsed:.2f}", tip_s=f"{tips_count/elapsed:.1f}", tips=tips_count)
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for job in jobs:
                if stop_event and stop_event.is_set():
                    break
                futures.append(ex.submit(_process_single_image_job, job))

            pbar = tqdm(as_completed(futures), total=len(futures), desc="Processing Images")
            for fut in pbar:
                if stop_event and stop_event.is_set():
                    print("Cancellation requested; waiting for running workers to finish...")
                    break

                result = fut.result()
                if result["missing_file"]:
                    missing_files.add(result["missing_file"])
                    continue
                if result["plug_fail"]:
                    plug_fail_count += 1
                    continue

                basename = result["basename"]
                for rid, dist in result["dists"].items():
                    if dist is not None:
                        length_cache[rid][basename] = dist

                elapsed = result["elapsed"]
                tips_count = result["tips_count"]
                pbar.set_postfix(frame_s=f"{elapsed:.2f}", tip_s=f"{tips_count/elapsed:.1f}", tips=tips_count)

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
    if plug_fail_count:
        print(f"Warning: Plug detection failed for {plug_fail_count} images.")
        
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
    parser.add_argument("--no-optimize", action="store_true", help="Disable ROI + early-stop optimizations")
    parser.add_argument("--roi-margin", type=int, default=220, help="ROI margin around plug and tips")
    parser.add_argument("--no-fallback", action="store_true", help="Disable full-frame fallback for failed ROI paths")
    parser.add_argument("--no-ends-opt", action="store_true", help="Disable MCP ends optimization")
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker processes (0=auto, 1=sequential)")
    args = parser.parse_args()
    compute_incremental_speeds(
        args.tracks,
        args.images,
        args.output,
        downscale=args.downscale,
        use_frangi=not args.fast,
        exponent=args.exponent,
        use_threshold=args.binary,
        optimize=not args.no_optimize,
        roi_margin=args.roi_margin,
        fallback_full_frame=not args.no_fallback,
        use_ends_optimization=not args.no_ends_opt,
        workers=args.workers,
    )
