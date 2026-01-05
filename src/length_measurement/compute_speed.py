
import json
import csv
import re
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

def parse_timestamp(filename):
    # Match 14 digits: YYYYMMDDHHMMSS
    # Example: A_03_5-20240924205359.jpg
    match = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", filename)
    if match:
        y, m, d, H, M, S = map(int, match.groups())
        return datetime(y, m, d, H, M, S)
    return None

def compute_speeds(input_file, output_csv):
    print(f"Loading measurements from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    # Organize by Root ID
    # roots[root_id] = [(timestamp, length, filename), ...]
    roots = defaultdict(list)
    
    for frame in data:
        filename = frame['image']
        ts = parse_timestamp(filename)
        if ts is None:
            print(f"Warning: Could not parse timestamp from {filename}")
            continue
            
        for m in frame['measurements']:
            rid = m['root_id']
            length = m['length_px']
            roots[rid].append((ts, length, filename))
            
    print(f"Processing {len(roots)} roots...")
    
    # Compute Speeds
    rows = []
    
    for rid, measurements in roots.items():
        # Sort by time
        measurements.sort(key=lambda x: x[0])
        
        # Calculate diffs
        for i in range(1, len(measurements)):
            t0, l0, f0 = measurements[i-1]
            t1, l1, f1 = measurements[i]
            
            # Time delta in hours
            dt_seconds = (t1 - t0).total_seconds()
            if dt_seconds <= 0:
                continue # Should not happen if sorted and distinct
                
            dt_hours = dt_seconds / 3600.0
            
            # Length delta
            dl = l1 - l0
            
            # Speed
            speed = dl / dt_hours
            
            rows.append({
                "root_id": rid,
                "image_prev": f0,
                "image_curr": f1,
                "time_prev": t0.isoformat(),
                "time_curr": t1.isoformat(),
                "dt_hours": round(dt_hours, 4),
                "length_prev_px": round(l0, 2),
                "length_curr_px": round(l1, 2),
                "growth_px": round(dl, 2),
                "speed_px_per_hour": round(speed, 2)
            })
            
    # Save CSV
    if not rows:
        print("No speeds computed. Check input data or timestamps.")
        return
        
    headers = [
        "root_id", "time_curr", "speed_px_per_hour", 
        "growth_px", "dt_hours", 
        "length_prev_px", "length_curr_px", 
        "time_prev", "image_prev", "image_curr"
    ]
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Saved {len(rows)} speed measurements to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute root growth speeds from length measurements.")
    parser.add_argument("--input", required=True, help="Path to root_lengths.json")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    
    args = parser.parse_args()
    compute_speeds(args.input, args.output)
