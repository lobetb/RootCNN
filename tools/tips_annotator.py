import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import json
import csv
import os
from pathlib import Path
import numpy as np

class RootTipAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Root Tip Annotator")
        self.root.geometry("1200x900")
        
        # Variables
        self.image_folder = None
        self.image_files = []
        self.current_image_idx = 0
        self.current_image = None
        self.original_image = None
        self.image_path = None
        
        # View settings
        self.patch_size = 800  # Size of displayed patch
        self.current_patch_x = 0
        self.current_patch_y = 0
        self.zoom_factor = 1.0
        
        # Mouse interaction
        self.drag_start_x = None
        self.drag_start_y = None
        self.dragging = False
        self.drag_start_patch_x = None
        self.drag_start_patch_y = None
        
        # Annotations
        self.annotations = {}  # Format: {image_filename: [(x, y), ...]}
        self.current_tips = []  # Tips for current image
        
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
        
        # Folder selection
        ttk.Button(control_frame, text="Select Folder", 
                  command=self.select_folder).pack(side=tk.LEFT, padx=(0, 10))
        
        # Image navigation
        ttk.Button(control_frame, text="Previous Image", 
                  command=self.prev_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Next Image", 
                  command=self.next_image).pack(side=tk.LEFT, padx=(0, 10))
        
        # Patch navigation (still useful for precise movement)
        ttk.Button(control_frame, text="←", 
                  command=self.move_left).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(control_frame, text="→", 
                  command=self.move_right).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(control_frame, text="↑", 
                  command=self.move_up).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(control_frame, text="↓", 
                  command=self.move_down).pack(side=tk.LEFT, padx=(0, 10))
        
        # Zoom controls and info
        ttk.Label(control_frame, text="Zoom:").pack(side=tk.LEFT)
        ttk.Button(control_frame, text="-", 
                  command=self.zoom_out).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Button(control_frame, text="+", 
                  command=self.zoom_in).pack(side=tk.LEFT, padx=(0, 10))
        
        self.zoom_label = ttk.Label(control_frame, text="1.0x")
        self.zoom_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Reset view
        ttk.Button(control_frame, text="Reset View", 
                  command=self.reset_view).pack(side=tk.LEFT, padx=(0, 10))
        
        # Undo last annotation
        ttk.Button(control_frame, text="Undo Last", 
                  command=self.undo_last).pack(side=tk.LEFT, padx=(0, 10))
        
        # Save options
        ttk.Button(control_frame, text="Save JSON", 
                  command=self.save_json).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(control_frame, text="Save CSV", 
                  command=self.save_csv).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="Select a folder to start")
        self.info_label.pack(side=tk.LEFT)
        
        self.coord_label = ttk.Label(info_frame, text="")
        self.coord_label.pack(side=tk.RIGHT)
        
        # Instructions
        instructions = ttk.Label(info_frame, 
                                text="Mouse wheel: zoom | Drag: pan | Click: add tip | Double-click tip in list: delete",
                                font=("Arial", 8))
        instructions.pack()
        
        # Canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas without scrollbars (we'll handle panning manually)
        self.canvas = tk.Canvas(canvas_frame, bg='gray', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Tips list frame
        tips_frame = ttk.LabelFrame(main_frame, text="Current Image Tips")
        tips_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Listbox for tips with scrollbar
        tips_list_frame = ttk.Frame(tips_frame)
        tips_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tips_listbox = tk.Listbox(tips_list_frame, height=4)
        tips_scrollbar = ttk.Scrollbar(tips_list_frame, orient=tk.VERTICAL, 
                                      command=self.tips_listbox.yview)
        self.tips_listbox.configure(yscrollcommand=tips_scrollbar.set)
        
        self.tips_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tips_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Delete selected tip button
        ttk.Button(tips_frame, text="Delete Selected", 
                  command=self.delete_selected_tip).pack(pady=(0, 5))
    
    def setup_bindings(self):
        # Mouse bindings
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # Mouse wheel bindings (different for different platforms)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down
        
        # Keyboard bindings
        self.root.bind("<Key>", self.on_key_press)
        self.root.focus_set()  # Enable key bindings
        
        # Make canvas focusable for mouse wheel events
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        
        # Double-click to delete tip
        self.tips_listbox.bind("<Double-Button-1>", lambda e: self.delete_selected_tip())
    
    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.image_folder = Path(folder)
            # Get all image files
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
            self.image_files = []
            for ext in extensions:
                self.image_files.extend(self.image_folder.glob(ext))
                self.image_files.extend(self.image_folder.glob(ext.upper()))
            
            self.image_files.sort()
            
            if self.image_files:
                self.current_image_idx = 0
                self.load_annotations()  # Try to load existing annotations
                self.load_current_image()
    
    def load_current_image(self):
        if not self.image_files:
            return
            
        self.image_path = self.image_files[self.current_image_idx]
        
        # Load image with OpenCV for large images
        self.original_image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            messagebox.showerror("Error", f"Could not load image: {self.image_path}")
            return
        
        # Reset view
        self.reset_view()
        
        # Load existing annotations for this image
        filename = self.image_path.name
        self.current_tips = self.annotations.get(filename, []).copy()
        
        self.update_display()
        self.update_info()
        self.update_tips_list()
    
    def reset_view(self):
        """Reset zoom and pan to show image from top-left"""
        self.current_patch_x = 0
        self.current_patch_y = 0
        self.zoom_factor = 1.0
        if self.original_image is not None:
            self.update_display()
    
    def get_canvas_size(self):
        """Get current canvas size"""
        self.canvas.update_idletasks()
        return self.canvas.winfo_width(), self.canvas.winfo_height()
    
    def update_display(self):
        if self.original_image is None:
            return
            
        try:
            canvas_w, canvas_h = self.get_canvas_size()
            h, w = self.original_image.shape
            
            # Calculate patch size based on canvas size
            display_w = canvas_w
            display_h = canvas_h
            
            # Calculate patch bounds in original image coordinates
            patch_w = max(1, int(display_w / self.zoom_factor))
            patch_h = max(1, int(display_h / self.zoom_factor))
            
            # Ensure patch stays within image bounds and convert to integers
            self.current_patch_x = max(0, min(int(self.current_patch_x), w - patch_w))
            self.current_patch_y = max(0, min(int(self.current_patch_y), h - patch_h))
            
            # Extract patch with proper integer indices
            end_x = min(self.current_patch_x + patch_w, w)
            end_y = min(self.current_patch_y + patch_h, h)
            
            patch = self.original_image[
                self.current_patch_y:end_y,
                self.current_patch_x:end_x
            ]
            
            # Handle empty patch case
            if patch.size == 0:
                return
            
            # Resize patch for display
            actual_patch_h, actual_patch_w = patch.shape
            
            # Calculate target display size
            target_w = max(1, int(actual_patch_w * self.zoom_factor))
            target_h = max(1, int(actual_patch_h * self.zoom_factor))
            
            # Limit maximum display size to prevent memory issues
            max_display_size = 4000
            if target_w > max_display_size or target_h > max_display_size:
                scale_factor = min(max_display_size / target_w, max_display_size / target_h)
                target_w = int(target_w * scale_factor)
                target_h = int(target_h * scale_factor)
            
            display_patch = cv2.resize(patch, (target_w, target_h), 
                                      interpolation=cv2.INTER_LINEAR if self.zoom_factor >= 1.0 else cv2.INTER_AREA)
            
            # Convert to RGB for PIL
            patch_rgb = cv2.cvtColor(display_patch, cv2.COLOR_GRAY2RGB)
            
            # Draw existing tips on the patch
            for tip_x, tip_y in self.current_tips:
                # Check if tip is in current patch
                rel_x = tip_x - self.current_patch_x
                rel_y = tip_y - self.current_patch_y
                
                if (0 <= rel_x < patch_w and 0 <= rel_y < patch_h):
                    # Scale coordinates for display
                    display_x = int(rel_x * self.zoom_factor)
                    display_y = int(rel_y * self.zoom_factor)
                    
                    # Ensure coordinates are within the display patch
                    if (0 <= display_x < display_patch.shape[1] and 
                        0 <= display_y < display_patch.shape[0]):
                        # Draw red circle with size based on zoom
                        radius = max(2, int(5 * self.zoom_factor))
                        thickness = max(1, int(2 * self.zoom_factor))
                        inner_radius = max(1, radius//2)
                        
                        cv2.circle(patch_rgb, (display_x, display_y), radius, (255, 0, 0), thickness)
                        cv2.circle(patch_rgb, (display_x, display_y), inner_radius, (255, 255, 0), -1)
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(patch_rgb)
            self.current_image = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
            
            # Update zoom label
            self.zoom_label.config(text=f"{self.zoom_factor:.1f}x")
            
        except Exception as e:
            print(f"Error in update_display: {e}")
            # Reset to safe state
            self.zoom_factor = max(0.1, self.zoom_factor)
            self.current_patch_x = max(0, int(self.current_patch_x))
            self.current_patch_y = max(0, int(self.current_patch_y))
    
    def on_drag_start(self, event):
        """Start dragging operation"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_start_patch_x = self.current_patch_x
        self.drag_start_patch_y = self.current_patch_y
        self.dragging = False  # We'll set this to True only if mouse moves
    
    def on_drag_motion(self, event):
        """Handle drag motion"""
        if self.drag_start_x is None or self.drag_start_y is None:
            return
            
        # Calculate drag distance
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        # Only start dragging if mouse moved significantly
        if not self.dragging and (abs(dx) > 3 or abs(dy) > 3):
            self.dragging = True
        
        if self.dragging and self.original_image is not None:
            # Convert screen drag to image coordinate drag
            image_dx = -dx / self.zoom_factor
            image_dy = -dy / self.zoom_factor
            
            # Update patch position
            new_patch_x = self.drag_start_patch_x + image_dx
            new_patch_y = self.drag_start_patch_y + image_dy
            
            # Constrain to image bounds
            canvas_w, canvas_h = self.get_canvas_size()
            h, w = self.original_image.shape
            
            patch_w = max(1, canvas_w / self.zoom_factor)
            patch_h = max(1, canvas_h / self.zoom_factor)
            
            self.current_patch_x = max(0, min(new_patch_x, w - patch_w))
            self.current_patch_y = max(0, min(new_patch_y, h - patch_h))
            
            self.update_display()
            self.update_info()
    
    def on_drag_end(self, event):
        """End dragging operation"""
        was_dragging = self.dragging
        self.drag_start_x = None
        self.drag_start_y = None
        self.dragging = False
        
        # Only register click if we weren't dragging
        if not was_dragging:
            self.add_tip_at_position(event.x, event.y)
    
    def on_click(self, event):
        """This is now handled by drag_end to distinguish from dragging"""
        pass
    
    def add_tip_at_position(self, canvas_x, canvas_y):
        """Add a tip at the given canvas position"""
        if self.original_image is None:
            return
            
        # Convert canvas coordinates to original image coordinates
        orig_x = int(self.current_patch_x + canvas_x / self.zoom_factor)
        orig_y = int(self.current_patch_y + canvas_y / self.zoom_factor)
        
        # Ensure coordinates are within image bounds
        h, w = self.original_image.shape
        if 0 <= orig_x < w and 0 <= orig_y < h:
            # Add tip
            self.current_tips.append((orig_x, orig_y))
            
            # Update annotations
            filename = self.image_path.name
            self.annotations[filename] = self.current_tips.copy()
            
            # Update display
            self.update_display()
            self.update_tips_list()
            
            print(f"Added tip at ({orig_x}, {orig_y})")
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel zoom"""
        if self.original_image is None:
            return
        
        try:
            # Get mouse position for zoom center
            mouse_x = event.x
            mouse_y = event.y
            
            # Calculate zoom factor change
            if hasattr(event, 'delta') and event.delta:  # Windows
                zoom_in = event.delta > 0
            else:  # Linux
                zoom_in = event.num == 4
            
            old_zoom = self.zoom_factor
            
            # Apply zoom with limits
            if zoom_in:
                self.zoom_factor = min(8.0, self.zoom_factor * 1.2)
            else:
                self.zoom_factor = max(0.05, self.zoom_factor / 1.2)  # Allow lower zoom
            
            # Zoom towards mouse position if zoom actually changed
            if abs(old_zoom - self.zoom_factor) > 0.001:
                # Calculate what point in the original image the mouse is pointing to
                old_image_x = self.current_patch_x + mouse_x / old_zoom
                old_image_y = self.current_patch_y + mouse_y / old_zoom
                
                # Calculate new patch position to keep mouse point fixed
                new_patch_x = old_image_x - mouse_x / self.zoom_factor
                new_patch_y = old_image_y - mouse_y / self.zoom_factor
                
                # Constrain to image bounds
                canvas_w, canvas_h = self.get_canvas_size()
                h, w = self.original_image.shape
                
                patch_w = max(1, canvas_w / self.zoom_factor)
                patch_h = max(1, canvas_h / self.zoom_factor)
                
                self.current_patch_x = max(0, min(new_patch_x, w - patch_w))
                self.current_patch_y = max(0, min(new_patch_y, h - patch_h))
                
                self.update_display()
                self.update_info()
                
        except Exception as e:
            print(f"Error in zoom: {e}")
            # Reset to safe zoom level
            self.zoom_factor = 1.0
            self.update_display()
    
    def on_mouse_move(self, event):
        """Show current coordinates"""
        if self.original_image is None or self.dragging:
            return
            
        # Convert canvas coordinates to original image coordinates
        orig_x = int(self.current_patch_x + event.x / self.zoom_factor)
        orig_y = int(self.current_patch_y + event.y / self.zoom_factor)
        
        # Show coordinates
        h, w = self.original_image.shape
        if 0 <= orig_x < w and 0 <= orig_y < h:
            self.coord_label.config(text=f"Position: ({orig_x}, {orig_y})")
        else:
            self.coord_label.config(text="Position: (out of bounds)")
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.keysym == 'Right':
            self.next_image()
        elif event.keysym == 'Left':
            self.prev_image()
        elif event.keysym == 'a':
            self.move_left()
        elif event.keysym == 'd':
            self.move_right()
        elif event.keysym == 'w':
            self.move_up()
        elif event.keysym == 's':
            self.move_down()
        elif event.keysym == 'z':
            self.undo_last()
        elif event.keysym == 'r':
            self.reset_view()
        elif event.keysym == 'equal' or event.keysym == 'plus':
            self.zoom_in()
        elif event.keysym == 'minus':
            self.zoom_out()
    
    def move_left(self):
        if self.original_image is not None:
            canvas_w, _ = self.get_canvas_size()
            step = canvas_w / 4 / self.zoom_factor  # Move 1/4 of screen
            self.current_patch_x = max(0, self.current_patch_x - step)
            self.update_display()
            self.update_info()
    
    def move_right(self):
        if self.original_image is not None:
            canvas_w, _ = self.get_canvas_size()
            step = canvas_w / 4 / self.zoom_factor
            w = self.original_image.shape[1]
            patch_w = canvas_w / self.zoom_factor
            self.current_patch_x = min(w - patch_w, self.current_patch_x + step)
            self.update_display()
            self.update_info()
    
    def move_up(self):
        if self.original_image is not None:
            _, canvas_h = self.get_canvas_size()
            step = canvas_h / 4 / self.zoom_factor
            self.current_patch_y = max(0, self.current_patch_y - step)
            self.update_display()
            self.update_info()
    
    def move_down(self):
        if self.original_image is not None:
            _, canvas_h = self.get_canvas_size()
            step = canvas_h / 4 / self.zoom_factor
            h = self.original_image.shape[0]
            patch_h = canvas_h / self.zoom_factor
            self.current_patch_y = min(h - patch_h, self.current_patch_y + step)
            self.update_display()
            self.update_info()
    
    def zoom_in(self):
        """Zoom in centered on canvas"""
        if self.original_image is None:
            return
        canvas_w, canvas_h = self.get_canvas_size()
        # Create a fake event at center for zoom_toward_point
        class FakeEvent:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.delta = 1
        
        center_event = FakeEvent(canvas_w//2, canvas_h//2)
        self.on_mouse_wheel(center_event)
    
    def zoom_out(self):
        """Zoom out centered on canvas"""
        if self.original_image is None:
            return
        canvas_w, canvas_h = self.get_canvas_size()
        class FakeEvent:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.delta = -1
        
        center_event = FakeEvent(canvas_w//2, canvas_h//2)
        self.on_mouse_wheel(center_event)
    
    def next_image(self):
        if self.image_files and self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self.load_current_image()
    
    def prev_image(self):
        if self.image_files and self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_current_image()
    
    def undo_last(self):
        if self.current_tips:
            removed_tip = self.current_tips.pop()
            filename = self.image_path.name
            self.annotations[filename] = self.current_tips.copy()
            self.update_display()
            self.update_tips_list()
            print(f"Removed tip at {removed_tip}")
    
    def delete_selected_tip(self):
        selection = self.tips_listbox.curselection()
        if selection:
            idx = selection[0]
            if 0 <= idx < len(self.current_tips):
                removed_tip = self.current_tips.pop(idx)
                filename = self.image_path.name
                self.annotations[filename] = self.current_tips.copy()
                self.update_display()
                self.update_tips_list()
                print(f"Deleted tip at {removed_tip}")
    
    def update_info(self):
        if self.image_files:
            filename = self.image_path.name
            progress = f"{self.current_image_idx + 1}/{len(self.image_files)}"
            h, w = self.original_image.shape
            patch_info = f"View: ({int(self.current_patch_x)}, {int(self.current_patch_y)})"
            tips_count = f"Tips: {len(self.current_tips)}"
            info = f"{filename} | {progress} | {w}x{h} | {patch_info} | {tips_count}"
            self.info_label.config(text=info)
    
    def update_tips_list(self):
        self.tips_listbox.delete(0, tk.END)
        for i, (x, y) in enumerate(self.current_tips):
            self.tips_listbox.insert(tk.END, f"{i+1}: ({x}, {y})")
    
    def load_annotations(self):
        """Try to load existing annotations from JSON file"""
        json_path = Path("annotations.json")
        if json_path.exists():
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            print(f"Loaded existing annotations from {json_path}")
    
    def save_json(self):
        if not self.image_folder:
            messagebox.showerror("Error", "No folder selected")
            return
            
        json_path = self.image_folder / "annotations.json"
        with open(json_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        total_tips = sum(len(tips) for tips in self.annotations.values())
        messagebox.showinfo("Success", f"Saved {len(self.annotations)} images with {total_tips} tips to {json_path}")
    
    def save_csv(self):
        if not self.image_folder:
            messagebox.showerror("Error", "No folder selected")
            return
            
        csv_path = self.image_folder / "annotations.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_filename', 'tip_x', 'tip_y'])
            
            for filename, tips in self.annotations.items():
                for x, y in tips:
                    writer.writerow([filename, x, y])
        
        total_tips = sum(len(tips) for tips in self.annotations.values())
        messagebox.showinfo("Success", f"Saved {len(self.annotations)} images with {total_tips} tips to {csv_path}")

def main():
    root = tk.Tk()
    app = RootTipAnnotator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
