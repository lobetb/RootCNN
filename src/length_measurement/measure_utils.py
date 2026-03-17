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

def find_support_boundary(image_arr, threshold=40, min_thickness=20):
    """
    Detects the Y-coordinate of the bottom edge of the main plant support.
    """
    if image_arr.ndim == 3:
        gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_arr
        
    row_vals = np.median(gray, axis=1)
    H = len(row_vals)
    
    bands = []
    current_start = None
    
    for y in range(H):
        if row_vals[y] < threshold:
            if current_start is None:
                current_start = y
        else:
            if current_start is not None:
                thickness = y - current_start
                bands.append((current_start, y, thickness))
                current_start = None
    
    if current_start is not None:
        bands.append((current_start, H, H - current_start))
        
    if not bands:
        return 0
        
    thickest_band = max(bands, key=lambda x: x[2])
    if thickest_band[2] >= min_thickness:
        return thickest_band[1] # Return the bottom Y
    return 0

def detect_plug(image_arr, y_boundary, black_threshold=50, white_threshold=200, min_consecutive=350):
    """
    Scan for a black plug (at least min_consecutive width) 200px below the support boundary.
    Returns (scan_y, center_x)
    """
    H, W = image_arr.shape[:2]
    scan_y = y_boundary + 200
    if scan_y >= H:
        return None, None
        
    if image_arr.ndim == 3:
        gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_arr
        
    line = gray[scan_y, :]
    
    # 1. Right-to-Left: find a clear zone first
    x = W - 1
    while x >= 0 and line[x] < white_threshold:
        x -= 1
    
    if x < 0:
        return None, None 
        
    # 2. Iteratively search for the plug
    while x >= 0:
        # Seek the start (right edge) of a black object
        while x >= 0 and line[x] > black_threshold:
            x -= 1
        
        if x < 0:
            break
            
        right_edge = x
        
        # Find the left edge of this black object
        while x >= 0 and line[x] < white_threshold:
            x -= 1
            
        left_edge = x + 1
        width = right_edge - left_edge
        
        if width >= min_consecutive:
            center_x = (left_edge + right_edge) / 2
            return scan_y, int(center_x)
            
    return None, None

def compute_cost_map(image_arr, prev_paths=None, active_ids=None, sigmas=None, alpha=0.01, use_frangi=True, exponent=4, use_threshold=False):
    """
    Compute the cost map for pathfinding.
    Combined cost = ((1 - Vesselness)**exponent) + alpha * (Distance to previous paths)
    """
    if sigmas is None:
        sigmas = range(1, 10, 2)
        
    if use_frangi:
        # 1. Vesselness Cost
        # Frangi returns [0, 1] response
        vesselness = frangi(image_arr, sigmas=sigmas, black_ridges=True)
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-6)
        
        if use_threshold:
            # Apply binary thresholding (Otsu) to vesselness
            v_uint8 = (vesselness * 255).astype(np.uint8)
            _, thresh = cv2.threshold(v_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            vesselness = thresh.astype(np.float32) / 255.0

        # Base cost: Background=1, Root=0
        # Apply exponent to penalize background pixels more heavily (non-linear cost)
        base_cost = (1.0 - vesselness)**exponent + 1e-4
    else:
        # Fast mode: Use raw inverted intensity
        if image_arr.max() > 1.0:
            norm_img = image_arr / 255.0
        else:
            norm_img = image_arr
            
        if use_threshold:
            # Apply binary thresholding to raw intensity
            img_uint8 = (norm_img * 255).astype(np.uint8)
            _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            norm_img = 1.0 - (thresh.astype(np.float32) / 255.0)

        # Apply exponent to raw intensity as well
        base_cost = norm_img**exponent + 1e-4
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
    return total_length, path_xy
