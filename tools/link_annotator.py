#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 11:53:55 2025

@author: ben
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import json
import csv
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import shutil

class RootTipTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Root Tip Tracker")
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
        self.tips1 = []  # Tips from first image
        self.tips2 = []  # Tips from second image
        
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
        
        # Linking
        self.selected_tip1 = None
        self.links = []  # Current pair links: [(tip1_idx, tip2_idx), ...]
        self.all_links = {}  # All links: {image1->image2: [(tip1_idx, tip2_idx), ...]}
        
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
        ttk.Button(file_frame, text="Save Annotations", 
                  command=self.save_annotations).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Import Links", 
                  command=self.import_existing_links).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Save Links", 
                  command=self.save_links).pack(side=tk.LEFT)
        
        # Navigation frame
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(nav_frame, text="Previous Pair", 
                  command=self.prev_pair).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Next Pair", 
                  command=self.next_pair).pack(side=tk.LEFT, padx=(0, 10))
        
        # Tip management frame
        tip_frame = ttk.Frame(control_frame)
        tip_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(tip_frame, text="Mode:").pack(side=tk.LEFT)
        self.tip_mode_var = tk.StringVar(value="link")
        ttk.Radiobutton(tip_frame, text="Link", variable=self.tip_mode_var, 
                       value="link").pack(side=tk.LEFT, padx=(5, 0))
        ttk.Radiobutton(tip_frame, text="Delete", variable=self.tip_mode_var, 
                       value="delete").pack(side=tk.LEFT, padx=(5, 0))
        
        # View controls
        view_frame = ttk.Frame(control_frame)
        view_frame.pack(side=tk.RIGHT)
        
        # Overlay controls
        ttk.Label(view_frame, text="Overlay:").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=0.5)
        alpha_scale = ttk.Scale(view_frame, from_=0.0, to=1.0, 
                               variable=self.alpha_var, orient=tk.HORIZONTAL,
                               length=100, command=self.on_alpha_change)
        alpha_scale.pack(side=tk.LEFT, padx=(5, 10))
        
        # Zoom controls
        ttk.Label(view_frame, text="Zoom:").pack(side=tk.LEFT)
        ttk.Button(view_frame, text="-", 
                  command=self.zoom_out).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Button(view_frame, text="+", 
                  command=self.zoom_in).pack(side=tk.LEFT, padx=(0, 5))
        
        self.zoom_label = ttk.Label(view_frame, text="1.0x")
        self.zoom_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Reset view
        ttk.Button(view_frame, text="Reset View", 
                  command=self.reset_view).pack(side=tk.LEFT)
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="Load annotations to start")
        self.info_label.pack(side=tk.LEFT)
        
        self.coord_label = ttk.Label(info_frame, text="")
        self.coord_label.pack(side=tk.RIGHT)
        
        # Folder info
        self.folder_label = ttk.Label(info_frame, text="", font=("Arial", 8), foreground="gray")
        self.folder_label.pack()
        
        # Instructions
        instructions = ttk.Label(info_frame, 
                                text="Mouse wheel: zoom | Drag: pan | Link mode: click red tip, then blue tip | Delete mode: click tip to remove",
                                font=("Arial", 8))
        instructions.pack()
        
        # Canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray', width=1000, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Links list frame
        links_frame = ttk.LabelFrame(main_frame, text="Current Links")
        links_frame.pack(fill=tk.X, pady=(10, 0))
        
        links_list_frame = ttk.Frame(links_frame)
        links_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.links_listbox = tk.Listbox(links_list_frame, height=6)
        links_scrollbar = ttk.Scrollbar(links_list_frame, orient=tk.VERTICAL, 
                                       command=self.links_listbox.yview)
        self.links_listbox.configure(yscrollcommand=links_scrollbar.set)
        
        self.links_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        links_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Delete selected link button
        ttk.Button(links_frame, text="Delete Selected Link", 
                  command=self.delete_selected_link).pack(pady=(0, 5))
    
    def setup_bindings(self):
        """Setup event bindings"""
        # Mouse events
        self.canvas.bind("<Button-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # Mouse wheel for zooming
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows/Mac
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux
        
        # Keyboard events
        self.canvas.bind("<Key>", self.on_key_press)
        self.canvas.focus_set()  # Enable keyboard events

    
    def load_annotations(self):
        """Load annotations from JSON file and images from images folder"""
        file_path = filedialog.askopenfilename(
            title="Select Annotations JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        # Store the file path for saving
        self.annotations_file_path = Path(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.annotations_data = json.load(f)
            
            if not self.annotations_data:
                messagebox.showerror("Error", "No annotations found in file")
                return
            
            # Set the folder containing the JSON file as base folder
            self.base_folder = Path(file_path).parent
            
            # Try default images folder first
            self.image_folder = self.base_folder / "images"
            if not self.image_folder.exists():
                if messagebox.askyesno("Image Folder Not Found", 
                                     f"Default images folder not found at: {self.image_folder}\nWould you like to select the image folder manually?"):
                    self.choose_image_folder()
                    return # choose_image_folder will handle the rest
                else:
                    return
            
            self.process_annotations()
            
            messagebox.showinfo("Success", 
                f"Loaded {len(self.annotations_data)} annotations from {len(self.image_files)} images\n"
                f"Found {len(self.image_pairs)} image pairs for tracking")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading annotations: {str(e)}")

    def choose_image_folder(self):
        """Manually choose the folder containing images"""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.image_folder = Path(folder)
            if hasattr(self, 'annotations_data') and self.annotations_data:
                self.process_annotations()
            else:
                messagebox.showinfo("Wait", "Please load annotations JSON first.")

    def process_annotations(self):
        """Process loaded annotations with the current image folder"""
        if not self.image_folder or not self.annotations_data:
            return

        # Get unique image names and sort them
        image_names = sorted(list(set(item['image'] for item in self.annotations_data)))
        
        # Verify that image files exist in the images folder
        self.image_files = []
        missing_images = []
        
        for image_name in image_names:
            image_path = self.image_folder / image_name
            if image_path.exists():
                self.image_files.append(image_path)
            else:
                # Fallback: check if the basename exists directly in image_folder
                basename = os.path.basename(image_name)
                basename_path = self.image_folder / basename
                if basename_path.exists():
                    self.image_files.append(basename_path)
                else:
                    missing_images.append(image_name)
        
        if missing_images:
            messagebox.showwarning("Warning", 
                f"The following images were not found in {self.image_folder}:\n" + 
                "\n".join(missing_images[:5]) + 
                (f"\n... and {len(missing_images)-5} more" if len(missing_images) > 5 else ""))
        
        if len(self.image_files) < 2:
            messagebox.showerror("Error", "Need at least 2 images for tracking")
            return
        
        # Create image pairs
        self.image_pairs = []
        for i in range(len(self.image_files) - 1):
            self.image_pairs.append((self.image_files[i], self.image_files[i + 1]))
        
        self.current_image_pair_idx = 0
        self.all_links = {}  # Reset links
        
        # Load first pair
        self.load_current_pair()
        self.update_info()

    def import_existing_links(self):
        """Import previously saved links from a JSON file"""
        file_path = filedialog.askopenfilename(
            title="Import Existing Links JSON",
            filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                imported_links = json.load(f)
            
            count = 0
            for pair_key, links in imported_links.items():
                if pair_key not in self.all_links:
                    self.all_links[pair_key] = []
                
                # Convert list of dicts to list of tuples
                for link in links:
                    if isinstance(link, dict):
                        tup = (link['tip1_index'], link['tip2_index'])
                    else:
                        tup = tuple(link)
                        
                    if tup not in self.all_links[pair_key]:
                        self.all_links[pair_key].append(tup)
                        count += 1
            
            # Reload current pair links if they exist in imported data
            if self.image1_path and self.image2_path:
                pair_key = f"{self.image1_path.name}->{self.image2_path.name}"
                if pair_key in self.all_links:
                    self.links = self.all_links[pair_key].copy()
                    self.update_links_list()
                    self.update_display()
            
            messagebox.showinfo("Success", f"Imported {count} new links from {file_path}")
            self.update_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import links: {str(e)}")
    
    def load_current_pair(self):
        """Load the current image pair and their tips"""
        if not self.image_pairs:
            return
        
        try:
            self.image1_path, self.image2_path = self.image_pairs[self.current_image_pair_idx]
            
            # Load images
            self.image1 = cv2.imread(str(self.image1_path), cv2.IMREAD_GRAYSCALE)
            self.image2 = cv2.imread(str(self.image2_path), cv2.IMREAD_GRAYSCALE)
            
            if self.image1 is None or self.image2 is None:
                messagebox.showerror("Error", "Could not load one or both images")
                return
            
            # Load tips for both images - handle relative paths if using subfolders
            try:
                # Try relative path matching first (standard for RootCNN_v3)
                rel1 = str(self.image1_path.relative_to(self.image_folder))
                rel2 = str(self.image2_path.relative_to(self.image_folder))
                
                self.tips1 = [item for item in self.annotations_data if item['image'] == rel1]
                self.tips2 = [item for item in self.annotations_data if item['image'] == rel2]
                
                # If no tips found with relative path, fallback to basename
                if not self.tips1:
                    self.tips1 = [item for item in self.annotations_data if item['image'] == self.image1_path.name]
                if not self.tips2:
                    self.tips2 = [item for item in self.annotations_data if item['image'] == self.image2_path.name]
                    
            except ValueError:
                # Fallback if image_folder is not parent of image_path
                image1_name = self.image1_path.name
                image2_name = self.image2_path.name
                self.tips1 = [item for item in self.annotations_data if item['image'] == image1_name]
                self.tips2 = [item for item in self.annotations_data if item['image'] == image2_name]
            
            # Use basename for pairs and display
            image1_name = self.image1_path.name
            image2_name = self.image2_path.name
            
            # Load existing links for this pair
            pair_key = f"{image1_name}->{image2_name}"
            self.links = self.all_links.get(pair_key, [])
            
            self.selected_tip1 = None
            self.reset_view()
            self.update_display()
            self.update_info()
            self.update_links_list()
            
            print(f"Loaded pair: {image1_name} ({len(self.tips1)} tips) -> {image2_name} ({len(self.tips2)} tips)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image pair: {str(e)}")
    
    def prev_pair(self):
        if self.image_pairs and self.current_image_pair_idx > 0:
            self.current_image_pair_idx -= 1
            self.load_current_pair()
    
    def next_pair(self):
        if self.image_pairs and self.current_image_pair_idx < len(self.image_pairs) - 1:
            self.current_image_pair_idx += 1
            self.load_current_pair()
    
    def reset_view(self):
        """Reset zoom and pan to show image from top-left"""
        self.current_patch_x = 0
        self.current_patch_y = 0
        self.zoom_factor = 1.0
        self.zoom_label.config(text="1.0x")
        if self.image1 is not None and self.image2 is not None:
            self.update_display()
    
    def get_canvas_size(self):
        """Get current canvas size"""
        self.canvas.update_idletasks()
        return self.canvas.winfo_width(), self.canvas.winfo_height()
    
    def update_display(self):
        if self.image1 is None or self.image2 is None:
            return
            
        try:
            canvas_w, canvas_h = self.get_canvas_size()
            h1, w1 = self.image1.shape
            h2, w2 = self.image2.shape
            h = max(h1, h2)
            w = max(w1, w2)
            
            # Calculate patch size based on canvas size
            display_w = canvas_w
            display_h = canvas_h
            
            # Calculate patch bounds in original image coordinates
            patch_w = max(1, int(display_w / self.zoom_factor))
            patch_h = max(1, int(display_h / self.zoom_factor))
            
            # Ensure patch stays within image bounds
            self.current_patch_x = max(0, min(int(self.current_patch_x), w - patch_w))
            self.current_patch_y = max(0, min(int(self.current_patch_y), h - patch_h))
            
            # Extract patches from both images
            end_x = min(self.current_patch_x + patch_w, w)
            end_y = min(self.current_patch_y + patch_h, h)
            
            # Handle different image sizes by padding if necessary
            patch1 = np.zeros((patch_h, patch_w), dtype=np.uint8)
            patch2 = np.zeros((patch_h, patch_w), dtype=np.uint8)
            
            # Extract from image1 if coordinates are within bounds
            if (self.current_patch_x < w1 and self.current_patch_y < h1):
                end_x1 = min(end_x, w1)
                end_y1 = min(end_y, h1)
                patch1[:end_y1-self.current_patch_y, :end_x1-self.current_patch_x] = \
                    self.image1[self.current_patch_y:end_y1, self.current_patch_x:end_x1]
            
            # Extract from image2 if coordinates are within bounds
            if (self.current_patch_x < w2 and self.current_patch_y < h2):
                end_x2 = min(end_x, w2)
                end_y2 = min(end_y, h2)
                patch2[:end_y2-self.current_patch_y, :end_x2-self.current_patch_x] = \
                    self.image2[self.current_patch_y:end_y2, self.current_patch_x:end_x2]
            
            # Resize patches for display
            actual_patch_h, actual_patch_w = patch1.shape
            target_w = max(1, int(actual_patch_w * self.zoom_factor))
            target_h = max(1, int(actual_patch_h * self.zoom_factor))
            
            # Limit maximum display size
            max_display_size = 4000
            if target_w > max_display_size or target_h > max_display_size:
                scale_factor = min(max_display_size / target_w, max_display_size / target_h)
                target_w = int(target_w * scale_factor)
                target_h = int(target_h * scale_factor)
            
            display_patch1 = cv2.resize(patch1, (target_w, target_h), 
                                       interpolation=cv2.INTER_LINEAR if self.zoom_factor >= 1.0 else cv2.INTER_AREA)
            display_patch2 = cv2.resize(patch2, (target_w, target_h), 
                                       interpolation=cv2.INTER_LINEAR if self.zoom_factor >= 1.0 else cv2.INTER_AREA)
            
            # Create overlay
            alpha = self.alpha_var.get()
            
            # Convert to RGB and apply color coding
            patch1_rgb = cv2.cvtColor(display_patch1, cv2.COLOR_GRAY2RGB)
            patch2_rgb = cv2.cvtColor(display_patch2, cv2.COLOR_GRAY2RGB)
            
            # Apply color tinting (red for image1, cyan for image2)
            patch1_colored = patch1_rgb.copy()
            patch1_colored[:, :, 1] = patch1_colored[:, :, 1] * 0.7  # Reduce green
            patch1_colored[:, :, 2] = patch1_colored[:, :, 2] * 0.7  # Reduce blue
            
            patch2_colored = patch2_rgb.copy()
            patch2_colored[:, :, 0] = patch2_colored[:, :, 0] * 0.7  # Reduce red
            
            # Blend the images
            overlay = cv2.addWeighted(patch1_colored, alpha, patch2_colored, 1 - alpha, 0)
            
            # Draw tips on the overlay
            self.draw_tips_on_patch(overlay, patch_w, patch_h)
            
            # Convert to PIL and display
            pil_image = Image.fromarray(overlay)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and add image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def draw_tips_on_patch(self, patch_rgb, patch_w, patch_h):
        """Draw tips on the RGB patch"""
        # Draw tips from image 1 (red)
        for i, tip in enumerate(self.tips1):
            tip_x, tip_y = tip['x'], tip['y']
            rel_x = tip_x - self.current_patch_x
            rel_y = tip_y - self.current_patch_y
            
            if (0 <= rel_x < patch_w and 0 <= rel_y < patch_h):
                display_x = int(rel_x * self.zoom_factor)
                display_y = int(rel_y * self.zoom_factor)
                
                if (0 <= display_x < patch_rgb.shape[1] and 0 <= display_y < patch_rgb.shape[0]):
                    # Adjust circle size based on zoom, with minimum size
                    base_radius = 8
                    radius = max(3, int(base_radius * max(0.5, self.zoom_factor)))
                    thickness = max(1, int(2 * max(0.5, self.zoom_factor)))
                    
                    # Highlight selected tip
                    color = (255, 255, 0) if i == self.selected_tip1 else (255, 100, 100)  # Yellow if selected, red otherwise
                    
                    cv2.circle(patch_rgb, (display_x, display_y), radius, color, thickness)
                    cv2.circle(patch_rgb, (display_x, display_y), max(1, radius//3), (255, 255, 255), -1)
                    
                    # Draw tip number - adjust font size based on zoom
                    font_scale = max(0.3, 0.4 * max(0.5, self.zoom_factor))
                    font_thickness = max(1, int(1 * max(0.5, self.zoom_factor)))
                    
                    cv2.putText(patch_rgb, str(i), 
                               (display_x + radius + 2, display_y - radius - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)
        
        # Draw tips from image 2 (blue) 
        for i, tip in enumerate(self.tips2):
            tip_x, tip_y = tip['x'], tip['y']
            rel_x = tip_x - self.current_patch_x
            rel_y = tip_y - self.current_patch_y
            
            if (0 <= rel_x < patch_w and 0 <= rel_y < patch_h):
                display_x = int(rel_x * self.zoom_factor)
                display_y = int(rel_y * self.zoom_factor)
                
                if (0 <= display_x < patch_rgb.shape[1] and 0 <= display_y < patch_rgb.shape[0]):
                    base_radius = 8
                    radius = max(3, int(base_radius * max(0.5, self.zoom_factor)))
                    thickness = max(1, int(2 * max(0.5, self.zoom_factor)))
                    
                    cv2.circle(patch_rgb, (display_x, display_y), radius, (100, 255, 255), thickness)
                    cv2.circle(patch_rgb, (display_x, display_y), max(1, radius//3), (255, 255, 255), -1)
                    
                    font_scale = max(0.3, 0.4 * max(0.5, self.zoom_factor))
                    font_thickness = max(1, int(1 * max(0.5, self.zoom_factor)))
                    
                    cv2.putText(patch_rgb, str(i), 
                               (display_x + radius + 2, display_y + radius + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
        
        # Draw existing links
        for tip1_idx, tip2_idx in self.links:
            if tip1_idx < len(self.tips1) and tip2_idx < len(self.tips2):
                tip1 = self.tips1[tip1_idx]
                tip2 = self.tips2[tip2_idx]
                
                # Check if both tips are in view
                rel_x1 = tip1['x'] - self.current_patch_x
                rel_y1 = tip1['y'] - self.current_patch_y
                rel_x2 = tip2['x'] - self.current_patch_x
                rel_y2 = tip2['y'] - self.current_patch_y
                
                if (0 <= rel_x1 < patch_w and 0 <= rel_y1 < patch_h and
                    0 <= rel_x2 < patch_w and 0 <= rel_y2 < patch_h):
                    
                    display_x1 = int(rel_x1 * self.zoom_factor)
                    display_y1 = int(rel_y1 * self.zoom_factor)
                    display_x2 = int(rel_x2 * self.zoom_factor)
                    display_y2 = int(rel_y2 * self.zoom_factor)
                    
                    line_thickness = max(1, int(2 * max(0.5, self.zoom_factor)))
                    cv2.line(patch_rgb, (display_x1, display_y1), (display_x2, display_y2), 
                            (0, 255, 0), line_thickness)
    
    def update_info(self):
        """Update info display"""
        if not self.image_pairs:
            self.info_label.config(text="Load annotations to start")
            return
        
        pair_num = self.current_image_pair_idx + 1
        total_pairs = len(self.image_pairs)
        
        image1_name = self.image1_path.name if self.image1_path else ""
        image2_name = self.image2_path.name if self.image2_path else ""
        
        num_tips1 = len(self.tips1)
        num_tips2 = len(self.tips2)
        num_links = len(self.links)
        
        mode = "DELETE" if self.tip_mode_var.get() == "delete" else "LINK"
        
        info_text = (f"Pair {pair_num}/{total_pairs} | Mode: {mode} | "
                    f"{image1_name} ({num_tips1} tips) -> {image2_name} ({num_tips2} tips) | "
                    f"Links: {num_links}")
        
        self.info_label.config(text=info_text)
        
        # Show folder path
        if hasattr(self, 'image_folder'):
            folder_text = f"Images from: {self.image_folder}"
            if hasattr(self, 'folder_label'):
                self.folder_label.config(text=folder_text)
    
    def update_links_list(self):
        """Update the links listbox"""
        self.links_listbox.delete(0, tk.END)
        for i, (tip1_idx, tip2_idx) in enumerate(self.links):
            if tip1_idx < len(self.tips1) and tip2_idx < len(self.tips2):
                tip1 = self.tips1[tip1_idx]
                tip2 = self.tips2[tip2_idx]
                link_text = f"{i+1}: Tip{tip1_idx}({tip1['x']},{tip1['y']}) -> Tip{tip2_idx}({tip2['x']},{tip2['y']})"
                self.links_listbox.insert(tk.END, link_text)
    
    def on_alpha_change(self, value):
        """Handle overlay alpha change"""
        self.overlay_alpha = float(value)
        self.update_display()
    
    def zoom_in(self):
        if self.zoom_factor < 10.0:
            self.zoom_factor = min(10.0, self.zoom_factor * 1.2)
            self.zoom_label.config(text=f"{self.zoom_factor:.2f}x")
            self.update_display()
    
    def zoom_out(self):
        if self.zoom_factor > 0.05:
            self.zoom_factor = max(0.05, self.zoom_factor / 1.2)
            self.zoom_label.config(text=f"{self.zoom_factor:.2f}x")
            self.update_display()
    
    # Mouse event handlers
    def on_drag_start(self, event):
        """Start dragging operation"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_start_patch_x = self.current_patch_x
        self.drag_start_patch_y = self.current_patch_y
        self.dragging = False
        print(f"Drag start at canvas: ({event.x}, {event.y})")
    
    def on_drag_motion(self, event):
        """Handle drag motion for panning"""
        if self.drag_start_x is None or self.drag_start_y is None:
            return
        
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        # Consider it dragging if moved more than a few pixels
        if abs(dx) > 5 or abs(dy) > 5:
            self.dragging = True
            
            # Update patch position (invert movement for natural panning)
            self.current_patch_x = self.drag_start_patch_x - (dx / self.zoom_factor)
            self.current_patch_y = self.drag_start_patch_y - (dy / self.zoom_factor)
            
            # Ensure within bounds
            if self.image1 is not None and self.image2 is not None:
                h = max(self.image1.shape[0], self.image2.shape[0])
                w = max(self.image1.shape[1], self.image2.shape[1])
                
                canvas_width = 800
                canvas_height = 600
                
                # Calculate visible area in image coordinates
                visible_width = canvas_width / self.zoom_factor
                visible_height = canvas_height / self.zoom_factor
                
                self.current_patch_x = max(0, min(self.current_patch_x, w - visible_width))
                self.current_patch_y = max(0, min(self.current_patch_y, h - visible_height))
            
            self.update_display()
    
    def on_drag_end(self, event):
        """End dragging operation and handle clicks"""
        was_dragging = self.dragging
        
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_start_patch_x = None
        self.drag_start_patch_y = None
        
        # Only process click if we weren't dragging
        if not was_dragging:
            self.on_click(event)
        
        self.dragging = False
    
    def on_click(self, event):
        """Handle canvas click events"""
        if self.image1 is None or self.image2 is None:
            print("No images loaded")
            return
        
        # Convert canvas coordinates to image coordinates
        image_x = self.current_patch_x + (event.x / self.zoom_factor)
        image_y = self.current_patch_y + (event.y / self.zoom_factor)
        
        print(f"Click at canvas: ({event.x}, {event.y}) -> image: ({image_x:.1f}, {image_y:.1f})")
        print(f"Mode: {self.tip_mode_var.get()}")
        
        mode = self.tip_mode_var.get()
        
        if mode == "delete":
            self.handle_delete_click(image_x, image_y)
        else:  # link mode
            self.handle_link_click(image_x, image_y)
    
    def handle_delete_click(self, image_x, image_y):
        """Handle click in delete mode"""
        # Adjust tolerance based on zoom level
        tolerance = max(15, 30 / max(0.1, self.zoom_factor))
        print(f"Delete mode - tolerance: {tolerance:.1f}")
        
        # Check for tips in image1
        for i, tip in enumerate(self.tips1):
            distance = ((tip['x'] - image_x) ** 2 + (tip['y'] - image_y) ** 2) ** 0.5
            print(f"Tip1 {i}: distance {distance:.1f} to ({tip['x']}, {tip['y']})")
            if distance < tolerance:
                self.delete_tip(1, i)
                return
        
        # Check for tips in image2
        for i, tip in enumerate(self.tips2):
            distance = ((tip['x'] - image_x) ** 2 + (tip['y'] - image_y) ** 2) ** 0.5
            print(f"Tip2 {i}: distance {distance:.1f} to ({tip['x']}, {tip['y']})")
            if distance < tolerance:
                self.delete_tip(2, i)
                return
        
        print("No tip found near click")
    
    def handle_link_click(self, image_x, image_y):
        """Handle click in link mode"""
        tolerance = max(15, 30 / max(0.1, self.zoom_factor))
        print(f"Link mode - tolerance: {tolerance:.1f}")
        
        # First, check if clicking on tip from image1 (red tips)
        for i, tip in enumerate(self.tips1):
            distance = ((tip['x'] - image_x) ** 2 + (tip['y'] - image_y) ** 2) ** 0.5
            print(f"Tip1 {i}: distance {distance:.1f} to ({tip['x']}, {tip['y']})")
            if distance < tolerance:
                self.selected_tip1 = i
                print(f"Selected tip {i} from first image")
                self.update_display()
                return
        
        # Then check tips from image2 (blue tips) - only if we have a selected tip1
        if self.selected_tip1 is not None:
            for i, tip in enumerate(self.tips2):
                distance = ((tip['x'] - image_x) ** 2 + (tip['y'] - image_y) ** 2) ** 0.5
                print(f"Tip2 {i}: distance {distance:.1f} to ({tip['x']}, {tip['y']})")
                if distance < tolerance:
                    # Create link
                    new_link = (self.selected_tip1, i)
                    if new_link not in self.links:
                        self.links.append(new_link)
                        print(f"Created link: tip {self.selected_tip1} -> tip {i}")
                        
                        # Store in all_links
                        pair_key = f"{self.image1_path.name}->{self.image2_path.name}"
                        self.all_links[pair_key] = self.links.copy()
                        
                        self.update_links_list()
                    else:
                        print("Link already exists")
                    
                    self.selected_tip1 = None
                    self.update_display()
                    return
        
        print("No tip found near click or no tip1 selected")

    
    def delete_tip(self, image_num, tip_index):
        """Delete a tip from the annotations and update links"""
        if image_num == 1:
            # Remove from tips1 and annotations_data
            deleted_tip = self.tips1.pop(tip_index)
            # Remove from original annotations_data
            self.annotations_data = [item for item in self.annotations_data 
                                   if not (item['image'] == self.image1_path.name and 
                                          item['x'] == deleted_tip['x'] and 
                                          item['y'] == deleted_tip['y'])]
            
            # Update existing links - remove links involving this tip and adjust indices
            new_links = []
            for tip1_idx, tip2_idx in self.links:
                if tip1_idx == tip_index:
                    continue  # Skip this link
                elif tip1_idx > tip_index:
                    new_links.append((tip1_idx - 1, tip2_idx))  # Adjust index
                else:
                    new_links.append((tip1_idx, tip2_idx))
            self.links = new_links
            
        else:  # image_num == 2
            # Remove from tips2 and annotations_data
            deleted_tip = self.tips2.pop(tip_index)
            # Remove from original annotations_data
            self.annotations_data = [item for item in self.annotations_data 
                                   if not (item['image'] == self.image2_path.name and 
                                          item['x'] == deleted_tip['x'] and 
                                          item['y'] == deleted_tip['y'])]
            
            # Update existing links - remove links involving this tip and adjust indices
            new_links = []
            for tip1_idx, tip2_idx in self.links:
                if tip2_idx == tip_index:
                    continue  # Skip this link
                elif tip2_idx > tip_index:
                    new_links.append((tip1_idx, tip2_idx - 1))  # Adjust index
                else:
                    new_links.append((tip1_idx, tip2_idx))
            self.links = new_links
        
        # Update all_links for this pair
        pair_key = f"{self.image1_path.name}->{self.image2_path.name}"
        self.all_links[pair_key] = self.links.copy()
        
        # Reset selection
        self.selected_tip1 = None
        
        # Update display
        self.update_display()
        self.update_info()
        self.update_links_list()
        
        print(f"Deleted tip {tip_index} from image {image_num}")
    
    def delete_selected_link(self):
        """Delete the selected link from the listbox"""
        selection = self.links_listbox.curselection()
        if selection:
            link_idx = selection[0]
            if 0 <= link_idx < len(self.links):
                deleted_link = self.links.pop(link_idx)
                print(f"Deleted link: {deleted_link}")
                
                # Update all_links
                pair_key = f"{self.image1_path.name}->{self.image2_path.name}"
                self.all_links[pair_key] = self.links.copy()
                
                self.update_links_list()
                self.update_display()
    
    def on_mouse_move(self, event):
        """Update coordinate display"""
        if self.image1 is None or self.image2 is None:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        image_x = int(self.current_patch_x + (canvas_x / self.zoom_factor))
        image_y = int(self.current_patch_y + (canvas_y / self.zoom_factor))
        
        self.coord_label.config(text=f"({image_x}, {image_y})")
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if self.image1 is None or self.image2 is None:
            return
        
        # Get mouse position before zoom
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        old_image_x = self.current_patch_x + (canvas_x / self.zoom_factor)
        old_image_y = self.current_patch_y + (canvas_y / self.zoom_factor)
        
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:  # Zoom in
            if self.zoom_factor < 10.0:
                self.zoom_factor = min(10.0, self.zoom_factor * 1.1)
        elif event.num == 5 or event.delta < 0:  # Zoom out
            if self.zoom_factor > 0.05:
                self.zoom_factor = max(0.05, self.zoom_factor / 1.1)
        
        # Adjust patch position to keep mouse point fixed
        new_image_x = self.current_patch_x + (canvas_x / self.zoom_factor)
        new_image_y = self.current_patch_y + (canvas_y / self.zoom_factor)
        
        self.current_patch_x += (old_image_x - new_image_x)
        self.current_patch_y += (old_image_y - new_image_y)
        
        self.zoom_label.config(text=f"{self.zoom_factor:.2f}x")
        self.update_display()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        key = event.keysym.lower()
        
        if key == 'left':
            self.prev_pair()
        elif key == 'right':
            self.next_pair()
        elif key == 'r':
            self.reset_view()
        elif key == 'escape':
            self.selected_tip1 = None
            self.update_display()
    
    def save_annotations(self):
        """Save modified annotations back to the original JSON file"""
        if not hasattr(self, 'annotations_file_path') or not self.annotations_file_path:
            messagebox.showerror("Error", "No annotations file loaded")
            return
        
        try:
            # Create backup
            backup_path = self.annotations_file_path.with_suffix(
                f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            shutil.copy2(self.annotations_file_path, backup_path)
            
            # Save modified annotations
            with open(self.annotations_file_path, 'w') as f:
                json.dump(self.annotations_data, f, indent=2)
            
            messagebox.showinfo("Success", 
                f"Annotations saved to {self.annotations_file_path}\n"
                f"Backup created: {backup_path.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")
    
    def save_links(self):
        """Save all links to a JSON file"""
        if not self.base_folder:
            messagebox.showerror("Error", "No base folder available")
            return
        
        if not self.all_links:
            messagebox.showwarning("Warning", "No links to save")
            return
        
        try:
            # Save current pair links first
            if self.image1_path and self.image2_path and self.links:
                pair_key = f"{self.image1_path.name}->{self.image2_path.name}"
                self.all_links[pair_key] = self.links.copy()
            
            links_path = self.base_folder / "tip_links.json"
            
            # Create formatted output
            formatted_links = {}
            for pair_key, links in self.all_links.items():
                if links:  # Only save pairs that have links
                    formatted_links[pair_key] = [{"tip1_index": t1, "tip2_index": t2} 
                                               for t1, t2 in links]
            
            with open(links_path, 'w') as f:
                json.dump(formatted_links, f, indent=2)
            
            total_links = sum(len(links) for links in formatted_links.values())
            messagebox.showinfo("Success", 
                f"Saved {total_links} links across {len(formatted_links)} image pairs to {links_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save links: {str(e)}")
    
    def move_left(self):
        self.current_patch_x -= 50 / self.zoom_factor
        self.current_patch_x = max(0, self.current_patch_x)
        self.update_display()
    
    def move_right(self):
        if self.image1 is not None and self.image2 is not None:
            w = max(self.image1.shape[1], self.image2.shape[1])
            self.current_patch_x += 50 / self.zoom_factor
            self.current_patch_x = min(self.current_patch_x, w)
            self.update_display()
    
    def move_up(self):
        self.current_patch_y -= 50 / self.zoom_factor
        self.current_patch_y = max(0, self.current_patch_y)
        self.update_display()
    
    def move_down(self):
        if self.image1 is not None and self.image2 is not None:
            h = max(self.image1.shape[0], self.image2.shape[0])
            self.current_patch_y += 50 / self.zoom_factor
            self.current_patch_y = min(self.current_patch_y, h)
            self.update_display()

def main():
    root = tk.Tk()
    app = RootTipTracker(root)
    root.mainloop()

if __name__ == "__main__":
    main()

