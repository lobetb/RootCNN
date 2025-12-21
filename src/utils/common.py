import torch
import re
import os
import numpy as np
from pathlib import Path

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_plant_id(filename):
    """
    Extracts plant ID from filename. 
    Format assumed: plantID-timestamp.extension or plantID_something-timestamp.extension
    Using part before the first dash as approved in the plan.
    """
    basename = os.path.basename(filename)
    if '-' in basename:
        return basename.split('-')[0]
    return "unknown"

def get_timestamp(filename):
    """Extracts yyyymmddhhmmss from filename."""
    match = re.search(r'(\d{14})', filename)
    return match.group(0) if match else filename

def discover_images(folder_path):
    """Recursively discover .jpg and .png images in a folder."""
    p = Path(folder_path)
    extensions = ("*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff")
    image_files = []
    for ext in extensions:
        image_files.extend(list(p.rglob(ext)))
    return sorted(image_files, key=lambda x: get_timestamp(x.name))

def filter_outlier_images(image_list, tip_features_dict, outlier_threshold=2.0):
    """
    Identifies images with an unusual number of tips compared to neighbors.
    image_list: list of sorted image basenames/paths.
    tip_features_dict: dict mapping basename -> list of tips.
    Returns: set of valid image basenames/paths.
    """
    if len(image_list) <= 3:
        return set(image_list)
        
    tip_counts = [len(tip_features_dict.get(img, [])) for img in image_list]
    valid_imgs = set(image_list)
    
    for i in range(len(tip_counts)):
        # Use a window of 5 (2 before, 2 after)
        start = max(0, i - 2)
        end = min(len(tip_counts), i + 3)
        neighbors = tip_counts[start:i] + tip_counts[i+1:end]
        if neighbors:
            median_neighbors = np.median(neighbors)
            if tip_counts[i] > outlier_threshold * median_neighbors and tip_counts[i] > 20:
                print(f"Filtering outlier image: {image_list[i]} (Tips: {tip_counts[i]}, Neighbor Median: {median_neighbors})")
                valid_imgs.remove(image_list[i])
    return valid_imgs
