import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_compound_image(img1_path, img2_path):
    """
    Create a compound image with img1 in Red channel and img2 in Blue channel.
    Green channel is set to 0.
    """
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        print(f"Error: Could not read {img1_path}")
        return None
    if img2 is None:
        print(f"Error: Could not read {img2_path}")
        return None

    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Create composite
    # B G R
    # img2 is t (Blue), img1 is t-1 (Red)
    # Actually, let's follow user request:
    # "image t and t-1 on top of each other... semi-transparent and a different color (i.e. blue and red)"
    # Standard composite:
    # R channel: img1 (t-1)
    # B channel: img2 (t)
    # G channel: 0
    # This results in magenta where they overlap perfectly.
    
    h, w = img1.shape
    compound = np.zeros((h, w, 3), dtype=np.uint8)
    compound[:, :, 2] = img1 # Red
    compound[:, :, 0] = img2 # Blue
    
    return compound

def draw_links(compound_img, links):
    """
    Draw yellow lines for links on the compound image.
    links is a list of dicts: {'tip1': {x, y, ...}, 'tip2': {x, y, ...}}
    """
    # Use copy to avoid modifying original if needed, but here we can modify in place
    out_img = compound_img.copy()
    
    for link in links:
        t1 = link['tip1']
        t2 = link['tip2']
        
        pt1 = (int(t1['x']), int(t1['y']))
        pt2 = (int(t2['x']), int(t2['y']))
        
        # Draw line
        # Yellow: (0, 255, 255) in BGR
        cv2.line(out_img, pt1, pt2, (0, 255, 255), 2)
        
        # Optional: Draw start and end points
        # Start (t-1) as Red dot? It's already red in channel r.
        # End (t) as Blue dot?
        # Let's draw small circles to make them visible
        cv2.circle(out_img, pt1, 3, (0, 0, 255), -1) # Red for t-1
        cv2.circle(out_img, pt2, 3, (255, 0, 0), -1) # Blue for t
        
    return out_img

def main():
    parser = argparse.ArgumentParser(description="Visual diagnostic for tip linking.")
    parser.add_argument("--links", default="output/positive_links.json", help="Path to positive_links.json")
    parser.add_argument("--images", required=True, help="Path to folder containing source images")
    parser.add_argument("--output", default="diagnostic_output", help="Output folder")
    
    args = parser.parse_args()
    
    links_data = load_json(args.links)
    img_folder = Path(args.images)
    out_folder = Path(args.output)
    out_folder.mkdir(exist_ok=True, parents=True)
    
    # Group links by (img1, img2) pair
    pairs = {}
    for link in links_data:
        img1_name = link['tip1']['image']
        img2_name = link['tip2']['image']
        
        key = (img1_name, img2_name)
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(link)
    
    print(f"Found {len(pairs)} unique image pairs.")
    
    for (img1_name, img2_name), current_links in tqdm(pairs.items(), desc="Generating Images"):
        # standard path resolution
        img1_path = img_folder / img1_name
        img2_path = img_folder / img2_name
        
        # Check if files exist
        if not img1_path.exists():
            # Try to find by basename as a fallback
            found = list(img_folder.rglob(Path(img1_name).name))
            if found: 
                img1_path = found[0]
            else:
                print(f"Warning: {img1_name} not found in {img_folder}")
                continue
                
        if not img2_path.exists():
             found = list(img_folder.rglob(Path(img2_name).name))
             if found: 
                 img2_path = found[0]
             else:
                print(f"Warning: {img2_name} not found in {img_folder}")
                continue
        
        compound = create_compound_image(img1_path, img2_path)
        if compound is None:
            continue
            
        viz_img = draw_links(compound, current_links)
        
        # Output filename: replace slashes with underscores to flatten output
        # or keep structure? Let's keep structure but ensure dir exists
        out_name = out_folder / f"viz_{img2_name}"
        out_name.parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(out_name), viz_img)
        if not success:
            print(f"Error: Failed to write {out_name}")
        
    print(f"Done. Diagnostic images saved to {out_folder}")

if __name__ == "__main__":
    main()
