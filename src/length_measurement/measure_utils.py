
import numpy as np
import cv2
import re
from datetime import datetime
from skimage.filters import frangi
from skimage.graph import MCP_Geometric

def parse_timestamp(filename):
    """
    Match 14 digits: YYYYMMDDHHMMSS
    Example: A_03_5-20240924205359.jpg
    """
    match = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", filename)
    if match:
        y, m, d, H, M, S = map(int, match.groups())
        return datetime(y, m, d, H, M, S)
    return None

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
    start_node = (int(start[1]), int(start[0]))
    end_node = (int(end[1]), int(end[0]))
    
    # Check bounds
    H, W = cost_map.shape
    if not (0 <= start_node[0] < H and 0 <= start_node[1] < W):
        return None, []
    if not (0 <= end_node[0] < H and 0 <= end_node[1] < W):
        return None, []

    mcp = MCP_Geometric(cost_map)
    try:
        cumulative_costs, traceback_map = mcp.find_costs(starts=[start_node], ends=[end_node])
        # traceback returns a list of (row, col) tuples
        path = mcp.traceback(end_node)
    except (ValueError, IndexError):
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
