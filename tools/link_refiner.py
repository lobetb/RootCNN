#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Link Refiner Tool
Based on link_annotator.py
Flags false positive links for fine-tuning.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import shutil

class LinkRefiner:
    def __init__(self, root):
        self.root = root
        self.root.title("Link Refiner - Flag False Positives")
        self.root.geometry("1400x1000")
        
        # Variables
        self.base_folder = None
        self.image_folder = None
        self.annotations_file_path = None
        self.annotations_data = []
        self.image_files = []
        self.image_pairs = []
        self.current_image_pair_idx = 0
        
        # Current images and tips
        self.image1 = None
        self.image2 = None
        self.image1_path = None
        self.image2_path = None
        self.tips1 = []  
        self.tips2 = []  
        
        # View settings
        self.current_patch_x = 0
        self.current_patch_y = 0
        self.zoom_factor = 1.0
        self.overlay_alpha = 0.5
        
        # Mouse interaction
        self.drag_start_x = None
        self.drag_start_y = None
        self.dragging = False
        self.drag_start_patch_x = None
        self.drag_start_patch_y = None
        
        # Linking & Refinement
        self.links = []  # List of (tip1_idx, tip2_idx)
        self.flagged_negatives = []  # List of indices in self.links that are flagged
        self.all_links = {}  # {pair_key: list of link dicts}
        self.all_flagged_negatives = {} # {pair_key: list of flagged link dicts}
        
        # UI Setup
        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File operations
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(file_frame, text="Load Annotations", 
                  command=self.load_annotations).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Choose Image Folder", 
                  command=self.choose_image_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Load Links", 
                  command=self.import_links).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Save Refined Data", 
                  command=self.save_refined_data).pack(side=tk.LEFT)
        
        # Navigation frame
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(nav_frame, text="Previous Pair", 
                  command=self.prev_pair).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Next Pair", 
                  command=self.next_pair).pack(side=tk.LEFT, padx=(0, 10))
        
        # View controls
        view_frame = ttk.Frame(control_frame)
        view_frame.pack(side=tk.RIGHT)
        
        ttk.Label(view_frame, text="Overlay:").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=0.5)
        alpha_scale = ttk.Scale(view_frame, from_=0.0, to=1.0, 
                               variable=self.alpha_var, orient=tk.HORIZONTAL,
                               length=100, command=self.on_alpha_change)
        alpha_scale.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(view_frame, text="Zoom:").pack(side=tk.LEFT)
        ttk.Button(view_frame, text="-", command=self.zoom_out).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Button(view_frame, text="+", command=self.zoom_in).pack(side=tk.LEFT, padx=(0, 5))
        self.zoom_label = ttk.Label(view_frame, text="1.0x")
        self.zoom_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="Load annotations and links to start")
        self.info_label.pack(side=tk.LEFT)
        
        self.coord_label = ttk.Label(info_frame, text="")
        self.coord_label.pack(side=tk.RIGHT)
        
        # Instructions
        instructions = ttk.Label(info_frame, 
                                text="Click a GREEN link to FLAG it as FALSE POSITIVE (Red) | Use wheel to zoom | Drag to pan",
                                font=("Arial", 9, "bold"))
        instructions.pack()
        
        # Canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray', width=1000, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Links list frame
        links_frame = ttk.LabelFrame(main_frame, text="Current Links (Green = Good, Red = Flagged Negative)")
        links_frame.pack(fill=tk.X, pady=(10, 0))
        
        links_list_frame = ttk.Frame(links_frame)
        links_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.links_listbox = tk.Listbox(links_list_frame, height=8)
        links_scrollbar = ttk.Scrollbar(links_list_frame, orient=tk.VERTICAL, 
                                       command=self.links_listbox.yview)
        self.links_listbox.configure(yscrollcommand=links_scrollbar.set)
        
        self.links_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        links_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_bindings(self):
        self.canvas.bind("<Button-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.root.bind("<Key>", self.on_key_press)

    def load_annotations(self):
        file_path = filedialog.askopenfilename(title="Select Annotations JSON File", filetypes=[("JSON files", "*.json")])
        if not file_path: return
        self.annotations_file_path = Path(file_path)
        self.base_folder = self.annotations_file_path.parent
        
        # Default image folder
        self.image_folder = self.base_folder / "images"
        
        try:
            with open(file_path, 'r') as f:
                self.annotations_data = json.load(f)
            
            self.process_images()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def choose_image_folder(self):
        """Manually choose the folder containing images"""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.image_folder = Path(folder)
            if self.annotations_data:
                self.process_images()
            else:
                messagebox.showinfo("Wait", "Please load annotations JSON first.")

    def process_images(self):
        """Process images and verify existence"""
        if not self.image_folder or not self.annotations_data:
            return

        image_names = sorted(list(set(item['image'] for item in self.annotations_data)))
        self.image_files = []
        missing = []
        
        for name in image_names:
            p = self.image_folder / name
            if p.exists():
                self.image_files.append(p)
            else:
                bp = self.image_folder / os.path.basename(name)
                if bp.exists():
                    self.image_files.append(bp)
                else:
                    missing.append(name)

        if missing and len(self.image_files) == 0:
            if messagebox.askyesno("Images Not Found", 
                                 f"Images not found in {self.image_folder}.\nWould you like to select the image folder manually?"):
                self.choose_image_folder()
                return

        if len(self.image_files) < 2:
            messagebox.showerror("Error", "Need at least 2 images for tracking")
            return
        
        self.image_pairs = [(self.image_files[i], self.image_files[i+1]) for i in range(len(self.image_files)-1)]
        self.current_image_pair_idx = 0
        self.load_current_pair()
        messagebox.showinfo("Success", f"Found {len(self.image_files)} images and {len(self.image_pairs)} pairs.")

    def import_links(self):
        file_path = filedialog.askopenfilename(title="Select Links JSON File", filetypes=[("JSON files", "*.json")])
        if not file_path: return
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Convert flat list (tracker output) to dict grouped by pairs
                new_all_links = {}
                for item in data:
                    if 'tip1' in item and 'tip2' in item:
                        img1 = os.path.basename(item['tip1']['image'])
                        img2 = os.path.basename(item['tip2']['image'])
                        pk = f"{img1}->{img2}"
                        if pk not in new_all_links: new_all_links[pk] = []
                        
                        # Add a compatible entry
                        new_all_links[pk].append({
                            'tip1_x': item['tip1']['x'],
                            'tip1_y': item['tip1']['y'],
                            'tip2_x': item['tip2']['x'],
                            'tip2_y': item['tip2']['y'],
                            'tip1_features': item['tip1'].get('features', []),
                            'tip2_features': item['tip2'].get('features', []),
                            'tip1_index': -1, # Will be resolved by coordinates
                            'tip2_index': -1
                        })
                self.all_links = new_all_links
            else:
                self.all_links = data
                
            self.load_current_pair()
            messagebox.showinfo("Success", "Links imported and grouped.")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")
            import traceback
            traceback.print_exc()

    def load_current_pair(self):
        if not self.image_pairs: return
        self.image1_path, self.image2_path = self.image_pairs[self.current_image_pair_idx]
        self.image1 = cv2.imread(str(self.image1_path), cv2.IMREAD_GRAYSCALE)
        self.image2 = cv2.imread(str(self.image2_path), cv2.IMREAD_GRAYSCALE)
        
        try:
            rel1 = str(self.image1_path.relative_to(self.image_folder))
            rel2 = str(self.image2_path.relative_to(self.image_folder))
        except ValueError:
            rel1 = self.image1_path.name
            rel2 = self.image2_path.name
            
        self.tips1 = [t for t in self.annotations_data if t['image'] == rel1 or t['image'] == self.image1_path.name]
        self.tips2 = [t for t in self.annotations_data if t['image'] == rel2 or t['image'] == self.image2_path.name]
        
        pair_key = f"{self.image1_path.name}->{self.image2_path.name}"
        raw_links = self.all_links.get(pair_key, [])
        
        self.links = []
        self.flagged_negatives = []
        
        # Coordinate matching if index is missing or -1
        coords1 = np.array([[t['x'], t['y']] for t in self.tips1]) if self.tips1 else np.array([]).reshape(0,2)
        coords2 = np.array([[t['x'], t['y']] for t in self.tips2]) if self.tips2 else np.array([]).reshape(0,2)
        
        for l in raw_links:
            idx1 = l.get('tip1_index', -1)
            idx2 = l.get('tip2_index', -1)
            
            if (idx1 == -1 or idx2 == -1) and 'tip1_x' in l:
                # Try coordinate match
                if len(coords1) > 0:
                    d1 = np.sqrt((coords1[:,0]-l['tip1_x'])**2 + (coords1[:,1]-l['tip1_y'])**2)
                    if np.min(d1) < 10: idx1 = int(np.argmin(d1))
                if len(coords2) > 0:
                    d2 = np.sqrt((coords2[:,0]-l['tip2_x'])**2 + (coords2[:,1]-l['tip2_y'])**2)
                    if np.min(d2) < 10: idx2 = int(np.argmin(d2))
                    
            if idx1 != -1 and idx2 != -1:
                self.links.append((idx1, idx2))
                if l.get('flagged_negative', False):
                    self.flagged_negatives.append(len(self.links) - 1)
        
        self.reset_view()
        self.update_display()
        self.update_info()
        self.update_links_list()

    def update_display(self):
        if self.image1 is None or self.image2 is None: return
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 10: canvas_w, canvas_h = 1000, 600
        
        patch_w, patch_h = int(canvas_w / self.zoom_factor), int(canvas_h / self.zoom_factor)
        h, w = max(self.image1.shape[0], self.image2.shape[0]), max(self.image1.shape[1], self.image2.shape[1])
        
        self.current_patch_x = max(0, min(self.current_patch_x, w - patch_w))
        self.current_patch_y = max(0, min(self.current_patch_y, h - patch_h))
        
        overlay = self.create_overlay(patch_w, patch_h)
        self.draw_links(overlay, patch_w, patch_h)
        
        pil_img = Image.fromarray(overlay)
        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def create_overlay(self, pw, ph):
        # Extract patches and blend
        p1 = np.zeros((ph, pw), dtype=np.uint8)
        p2 = np.zeros((ph, pw), dtype=np.uint8)
        
        px, py = int(self.current_patch_x), int(self.current_patch_y)
        ex, ey = px + pw, py + ph
        h1, w1 = self.image1.shape
        h2, w2 = self.image2.shape
        
        if px < w1 and py < h1:
            ex1, ey1 = min(ex, w1), min(ey, h1)
            p1[:ey1-py, :ex1-px] = self.image1[py:ey1, px:ex1]
        if px < w2 and py < h2:
            ex2, ey2 = min(ex, w2), min(ey, h2)
            p2[:ey2-py, :ex2-px] = self.image2[py:ey2, px:ex2]
            
        p1_rgb = cv2.cvtColor(cv2.resize(p1, (int(pw*self.zoom_factor), int(ph*self.zoom_factor))), cv2.COLOR_GRAY2RGB)
        p2_rgb = cv2.cvtColor(cv2.resize(p2, (int(pw*self.zoom_factor), int(ph*self.zoom_factor))), cv2.COLOR_GRAY2RGB)
        
        # Color tinting
        p1_rgb[:,:,1] = p1_rgb[:,:,1] * 0.7; p1_rgb[:,:,2] = p1_rgb[:,:,2] * 0.7 # Reddish
        p2_rgb[:,:,0] = p2_rgb[:,:,0] * 0.7 # Cyanic
        
        return cv2.addWeighted(p1_rgb, self.alpha_var.get(), p2_rgb, 1-self.alpha_var.get(), 0)

    def draw_links(self, overlay, pw, ph):
        for i, (idx1, idx2) in enumerate(self.links):
            if idx1 < len(self.tips1) and idx2 < len(self.tips2):
                t1, t2 = self.tips1[idx1], self.tips2[idx2]
                x1, y1 = int((t1['x'] - self.current_patch_x) * self.zoom_factor), int((t1['y'] - self.current_patch_y) * self.zoom_factor)
                x2, y2 = int((t2['x'] - self.current_patch_x) * self.zoom_factor), int((t2['y'] - self.current_patch_y) * self.zoom_factor)
                
                color = (0, 0, 255) if i in self.flagged_negatives else (0, 255, 0)
                thickness = 3 if i in self.flagged_negatives else 2
                cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(overlay, (x1, y1), 4, color, -1)
                cv2.circle(overlay, (x2, y2), 4, color, -1)

    def on_click(self, event):
        # Find if click is near a link
        img_x = self.current_patch_x + (event.x / self.zoom_factor)
        img_y = self.current_patch_y + (event.y / self.zoom_factor)
        
        best_link = -1
        min_dist = 20 # pixels tolerance
        
        for i, (idx1, idx2) in enumerate(self.links):
            if idx1 < len(self.tips1) and idx2 < len(self.tips2):
                t1, t2 = self.tips1[idx1], self.tips2[idx2]
                # Distance from point to line segment
                p = np.array([img_x, img_y])
                a = np.array([t1['x'], t1['y']])
                b = np.array([t2['x'], t2['y']])
                
                # Projection
                l2 = np.sum((a-b)**2)
                if l2 == 0: dist = np.linalg.norm(p-a)
                else:
                    t = max(0, min(1, np.dot(p-a, b-a) / l2))
                    projection = a + t * (b-a)
                    dist = np.linalg.norm(p - projection)
                
                if dist < min_dist:
                    min_dist = dist
                    best_link = i
        
        if best_link != -1:
            if best_link in self.flagged_negatives:
                self.flagged_negatives.remove(best_link)
            else:
                self.flagged_negatives.append(best_link)
            self.update_display()
            self.update_links_list()
            self.update_info()

    def save_refined_data(self):
        if not self.all_links: return
        
        # Sync current pair (add features natively before saving)
        # We need to make sure features are present in the output.
        # all_links holds the dict of links.
        for pair_key, raw_links in self.all_links.items():
            img1, img2 = pair_key.split("->")
            # Get features for these images from annotations_data
            tips1 = [t for t in self.annotations_data if os.path.basename(t['image']) == img1]
            tips2 = [t for t in self.annotations_data if os.path.basename(t['image']) == img2]
            
            coords1 = np.array([[t['x'], t['y']] for t in tips1]) if tips1 else np.array([]).reshape(0,2)
            coords2 = np.array([[t['x'], t['y']] for t in tips2]) if tips2 else np.array([]).reshape(0,2)
            
            for i, l in enumerate(raw_links):
                # Update current pair's memory of flags
                if pair_key == f"{os.path.basename(self.image1_path)}->{os.path.basename(self.image2_path)}":
                    l['flagged_negative'] = (i in self.flagged_negatives)
                
                # Fetch features if missing
                if not l.get('tip1_features') and len(coords1) > 0:
                    d1 = np.sqrt((coords1[:,0]-l['tip1_x'])**2 + (coords1[:,1]-l['tip1_y'])**2)
                    if np.min(d1) < 10: 
                        idx = int(np.argmin(d1))
                        l['tip1_features'] = tips1[idx].get('features', [])
                
                if not l.get('tip2_features') and len(coords2) > 0:
                    d2 = np.sqrt((coords2[:,0]-l['tip2_x'])**2 + (coords2[:,1]-l['tip2_y'])**2)
                    if np.min(d2) < 10: 
                        idx = int(np.argmin(d2))
                        l['tip2_features'] = tips2[idx].get('features', [])
            
        file_path = self.base_folder / "refined_links.json"
        with open(file_path, 'w') as f:
            json.dump(self.all_links, f, indent=2)
        
        # Also export a specific negatives file for the trainer
        negatives = {}
        for pk, links in self.all_links.items():
            negs = [l for l in links if l.get('flagged_negative', False)]
            if negs: negatives[pk] = negs
            
        neg_path = self.base_folder / "flagged_negatives.json"
        with open(neg_path, 'w') as f:
            json.dump(negatives, f, indent=2)
            
        messagebox.showinfo("Success", f"Saved refined data to {file_path}\nFlagged negatives exported to {neg_path}")

    # Standard handlers (adapted from link_annotator.py)
    def on_drag_start(self, e): self.drag_start_x, self.drag_start_y = e.x, e.y; self.drag_start_patch_x, self.drag_start_patch_y = self.current_patch_x, self.current_patch_y; self.dragging = False
    def on_drag_motion(self, e):
        dx, dy = e.x - self.drag_start_x, e.y - self.drag_start_y
        if abs(dx) > 5 or abs(dy) > 5:
            self.dragging = True
            self.current_patch_x, self.current_patch_y = self.drag_start_patch_x - (dx/self.zoom_factor), self.drag_start_patch_y - (dy/self.zoom_factor)
            self.update_display()
    def on_drag_end(self, e):
        if not self.dragging: self.on_click(e)
        self.dragging = False
    def on_mouse_move(self, e): self.coord_label.config(text=f"({int(self.current_patch_x + e.x/self.zoom_factor)}, {int(self.current_patch_y + e.y/self.zoom_factor)})")
    def on_mouse_wheel(self, e):
        factor = 1.1 if (e.num==4 or (hasattr(e, 'delta') and e.delta > 0)) else 0.9
        self.zoom_factor *= factor
        self.zoom_label.config(text=f"{self.zoom_factor:.2f}x")
        self.update_display()
    def on_key_press(self, e):
        k = e.keysym.lower()
        if k == 'left': self.prev_pair()
        elif k == 'right': self.next_pair()
    def prev_pair(self):
        if self.current_image_pair_idx > 0: self.current_image_pair_idx -= 1; self.load_current_pair()
    def next_pair(self):
        if self.current_image_pair_idx < len(self.image_pairs)-1: self.current_image_pair_idx += 1; self.load_current_pair()
    def reset_view(self): self.current_patch_x = 0; self.current_patch_y = 0; self.zoom_factor = 1.0; self.zoom_label.config(text="1.0x")
    def on_alpha_change(self, v): self.update_display()
    def zoom_in(self): self.zoom_factor *= 1.2; self.update_display(); self.zoom_label.config(text=f"{self.zoom_factor:.2f}x")
    def zoom_out(self): self.zoom_factor /= 1.2; self.update_display(); self.zoom_label.config(text=f"{self.zoom_factor:.2f}x")
    def update_info(self):
        pk = f"{self.image1_path.name}->{self.image2_path.name}" if self.image1_path else ""
        self.info_label.config(text=f"Pair {self.current_image_pair_idx+1}/{len(self.image_pairs)} | {pk} | Links: {len(self.links)} | Flagged: {len(self.flagged_negatives)}")
    def update_links_list(self):
        self.links_listbox.delete(0, tk.END)
        for i, (idx1, idx2) in enumerate(self.links):
            status = "[BAD]" if i in self.flagged_negatives else "[OK]"
            self.links_listbox.insert(tk.END, f"{status} Link {i}: {idx1} -> {idx2}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LinkRefiner(root)
    root.mainloop()
