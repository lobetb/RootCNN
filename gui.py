import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import sys
import os
import queue
import subprocess
from pathlib import Path

# Fix python path to allow imports from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.core import train_detection, export_features_for_folder
from src.association.core import train_linker, associate_tips_multi_plant
from src.length_measurement.compute_speed import compute_incremental_speeds


class ThreadSafeConsole:
    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root
        self.queue = queue.Queue()
        self.update_interval = 250
        self.root.after(self.update_interval, self.process_queue)

    def write(self, string):
        self.queue.put(string)

    def flush(self):
        pass

    def process_queue(self):
        while not self.queue.empty():
            try:
                msg = self.queue.get_nowait()
                if '\r' in msg:
                    parts = msg.split('\r')
                    for i, part in enumerate(parts):
                        if i > 0:
                            self.text_widget.delete("end-1c linestart", "end-1c")
                        self.text_widget.insert(tk.END, part)
                else:
                    self.text_widget.insert(tk.END, msg)
                self.text_widget.see(tk.END)
            except queue.Empty:
                break
        self.root.after(self.update_interval, self.process_queue)

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        tw = self.tip_window = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

class RootCNN_V2_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RootCNN")
        self.root.geometry("900x700")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.stop_event = threading.Event()

        # Tabs
        self.create_annotation_tools_tab()
        self.create_detection_train_tab()
        self.create_detect_tips_tab()
        self.create_linker_train_tab()
        self.create_association_tab()
        self.create_growth_speed_tab()

        # Console Output
        self.console_frame = ttk.LabelFrame(root, text="Console Output")
        self.console_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.console_text = tk.Text(self.console_frame, height=12)
        self.console_text.pack(fill='both', expand=True)
        
        self.console = ThreadSafeConsole(self.console_text, self.root)
        sys.stdout = self.console
        sys.stderr = self.console

    def browse_dir(self, entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def browse_file(self, entry, save=False):
        if save:
            path = filedialog.asksaveasfilename()
        else:
            path = filedialog.askopenfilename()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def add_info_icon(self, parent, row, column, text):
        info_icon = ttk.Label(parent, text="ⓘ", cursor="hand2", foreground="blue")
        info_icon.grid(row=row, column=column, padx=2, sticky='w')
        ToolTip(info_icon, text)
        return info_icon

    def run_wrapper(self, start_btn, stop_btn, func, *args, **kwargs):
        def task():
            start_btn.config(state=tk.DISABLED)
            stop_btn.config(state=tk.NORMAL)
            self.stop_event.clear()
            
            # Inject stop_event into kwargs
            kwargs['stop_event'] = self.stop_event
            
            try:
                func(*args, **kwargs)
                if not self.stop_event.is_set():
                    messagebox.showinfo("Success", "Operation completed successfully!")
                else:
                    messagebox.showinfo("Cancelled", "Operation was cancelled.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
            finally:
                start_btn.config(state=tk.NORMAL)
                stop_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=task, daemon=True).start()

    def stop_execution(self):
        self.stop_event.set()
        print("\n[STOP] Cancellation requested...")

    # --- TAB: Annotation Tools ---
    def create_annotation_tools_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="0. Annotation Tools")
        
        row = 0
        ttk.Label(frame, text="Use these tools to generate training data for the pipeline.", font=("Helvetica", 11, "italic")).grid(row=row, column=0, columnspan=3, sticky='w', padx=5, pady=10)

        row += 1
        ttk.Label(frame, text="1. Tips Annotator:", font=("Helvetica", 10, "bold")).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        ttk.Label(frame, text="Manually annotate root tips in images for training the detector (Step 1).").grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Button(frame, text="Launch Tips Annotator", command=lambda: self.launch_script("tools/tips_annotator.py")).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="2. Link Annotator:", font=("Helvetica", 10, "bold")).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        ttk.Label(frame, text="Manually annotate links between tips across consecutive frames for training the linker (Step 3).").grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Button(frame, text="Launch Link Annotator", command=lambda: self.launch_script("tools/link_annotator.py")).grid(row=row, column=2, padx=5, pady=5)

    def launch_script(self, script_path):
        script_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_path)
        if not os.path.exists(script_abs_path):
            messagebox.showerror("Error", f"Script not found: {script_path}")
            return
        
        print(f"Launching {script_path}...")
        try:
            # We use the same python interpreter that is currently running
            subprocess.Popen([sys.executable, script_abs_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch script: {str(e)}")

    # --- TAB: Detection Training ---
    def create_detection_train_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="1. Train Detector")
        
        row = 0
        ttk.Label(frame, text="Training Images Folder:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.det_train_img_entry = ttk.Entry(frame, width=50)
        self.det_train_img_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_dir(self.det_train_img_entry)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Annotations JSON:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.det_train_ann_entry = ttk.Entry(frame, width=50)
        self.det_train_ann_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.det_train_ann_entry)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Epochs:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.det_epochs_entry = ttk.Entry(frame, width=10)
        self.det_epochs_entry.insert(0, "20")
        self.det_epochs_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Training duration. 20-50 epochs recommended for new data.")

        row += 1
        ttk.Label(frame, text="Batch Size:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.det_batch_entry = ttk.Entry(frame, width=10)
        self.det_batch_entry.insert(0, "4")
        self.det_batch_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Number of images processed per step. Decrease if out of GPU memory.")

        row += 1
        ttk.Label(frame, text="Output Model Name:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.det_model_name_entry = ttk.Entry(frame, width=30)
        self.det_model_name_entry.insert(0, "detector.pth")
        self.det_model_name_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Log File (optional):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.det_train_log_entry = ttk.Entry(frame, width=50)
        self.det_train_log_entry.insert(0, "output/logs/detection_training.json")
        self.det_train_log_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.det_train_log_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        self.det_start_btn = ttk.Button(frame, text="Start Detector Training", command=self.run_det_train)
        self.det_start_btn.grid(row=row, column=0, pady=20)
        self.det_stop_btn = ttk.Button(frame, text="Stop Execution", command=self.stop_execution, state=tk.DISABLED)
        self.det_stop_btn.grid(row=row, column=1, pady=20)

    def run_det_train(self):
        img_dir = self.det_train_img_entry.get()
        ann_file = self.det_train_ann_entry.get()
        epochs = int(self.det_epochs_entry.get())
        bs = int(self.det_batch_entry.get())
        mname = self.det_model_name_entry.get().strip()
        log_file = self.det_train_log_entry.get().strip() or None

        if not img_dir or not ann_file:
            messagebox.showwarning("Input Required", "Please provide image folder and annotations.")
            return

        self.run_wrapper(self.det_start_btn, self.det_stop_btn, train_detection, img_dir, ann_file, epochs=epochs, batch_size=bs, model_name=mname, log_file=log_file)

    # --- TAB: Detect Tips ---
    def create_detect_tips_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="2. Detect Tips")

        row = 0
        ttk.Label(frame, text="Images Folder (Recursive):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.exp_img_entry = ttk.Entry(frame, width=50)
        self.exp_img_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_dir(self.exp_img_entry)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Detector Model Checkpoint:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.exp_model_entry = ttk.Entry(frame, width=50)
        self.exp_model_entry.insert(0, "models/detector.pth")
        self.exp_model_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.exp_model_entry)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Output Features JSON:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.exp_json_entry = ttk.Entry(frame, width=50)
        self.exp_json_entry.insert(0, "output/deep_tip_features.json")
        self.exp_json_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.exp_json_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Primary output containing tip coordinates and deep features.")

        row += 1
        self.exp_extract_feat_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Export deep features (necessary for tracking)", variable=self.exp_extract_feat_var).grid(row=row, column=0, columnspan=2, sticky='w', padx=5)
        self.add_info_icon(frame, row, 3, "If unchecked, only tip coordinates will be saved. Deep features are needed for linking.")

        row += 1
        self.exp_use_gt_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Use Ground Truth Tips?", variable=self.exp_use_gt_var).grid(row=row, column=0, columnspan=2, sticky='w', padx=5)
        self.add_info_icon(frame, row, 3, "Use manual annotations instead of detector predictions.")

        row += 1
        ttk.Label(frame, text="GT Annotations (if used):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.exp_ann_entry = ttk.Entry(frame, width=50)
        self.exp_ann_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.exp_ann_entry)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Detection Score Threshold:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.exp_thresh_entry = ttk.Entry(frame, width=10)
        self.exp_thresh_entry.insert(0, "0.5")
        self.exp_thresh_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Confidence threshold for tip detection (0.0 to 1.0).")

        row += 1
        ttk.Label(frame, text="Detection log file:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.exp_log_entry = ttk.Entry(frame, width=50)
        self.exp_log_entry.insert(0, "output/logs/detection.json")
        self.exp_log_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.exp_log_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Left Margin Exclusion (px):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.exp_margin_left_entry = ttk.Entry(frame, width=10)
        self.exp_margin_left_entry.insert(0, "0")
        self.exp_margin_left_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Pixels from the left edge to ignore when detecting tips.")

        row += 1
        ttk.Label(frame, text="Right Margin Exclusion (px):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.exp_margin_right_entry = ttk.Entry(frame, width=10)
        self.exp_margin_right_entry.insert(0, "0")
        self.exp_margin_right_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Pixels from the right edge to ignore when detecting tips.")

        row += 1
        self.exp_start_btn = ttk.Button(frame, text="Run Detection & Feature Extraction", command=self.run_feature_export)
        self.exp_start_btn.grid(row=row, column=0, pady=20)
        self.exp_stop_btn = ttk.Button(frame, text="Stop Execution", command=self.stop_execution, state=tk.DISABLED)
        self.exp_stop_btn.grid(row=row, column=1, pady=20)

    def run_feature_export(self):
        img_folder = self.exp_img_entry.get()
        model_ckpt = self.exp_model_entry.get()
        out_json = self.exp_json_entry.get()
        use_gt = self.exp_use_gt_var.get()
        extract_feat = self.exp_extract_feat_var.get()
        thresh = float(self.exp_thresh_entry.get())
        ann = self.exp_ann_entry.get()
        log_file = self.exp_log_entry.get().strip() or None
        margin_left = int(self.exp_margin_left_entry.get())
        margin_right = int(self.exp_margin_right_entry.get())

        if not img_folder or not model_ckpt:
            messagebox.showwarning("Input Required", "Please provide image folder and model.")
            return

        self.run_wrapper(self.exp_start_btn, self.exp_stop_btn, export_features_for_folder, img_folder, model_ckpt, out_json, annotations_json=ann, use_gt=use_gt, extract_features=extract_feat, threshold=thresh, margin_left=margin_left, margin_right=margin_right, log_file=log_file)

    # --- TAB: Linker Training ---
    def create_linker_train_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="3. Train Linker")

        row = 0
        ttk.Label(frame, text="Link Annotations JSON:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.link_ann_entry = ttk.Entry(frame, width=50)
        self.link_ann_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.link_ann_entry)).grid(row=row, column=2, padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Unified file from Link Annotator (tip_links.json) containing features and coordinates.")

        row += 1
        ttk.Label(frame, text="Epochs:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.link_epochs_entry = ttk.Entry(frame, width=10)
        self.link_epochs_entry.insert(0, "20")
        self.link_epochs_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Output Model Name:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.link_model_entry = ttk.Entry(frame, width=30)
        self.link_model_entry.insert(0, "gnn_linker.pth")
        self.link_model_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)

        row += 1
        self.link_use_gnn_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Use GNN Architecture (Recommended)", variable=self.link_use_gnn_var).grid(row=row, column=0, columnspan=2, sticky='w', padx=5)
        self.add_info_icon(frame, row, 3, "Graph Neural Network often provides better temporal consistency.")

        row += 1
        ttk.Label(frame, text="Log File (optional):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.link_train_log_entry = ttk.Entry(frame, width=50)
        self.link_train_log_entry.insert(0, "output/logs/linker_training.json")
        self.link_train_log_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.link_train_log_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        self.link_start_btn = ttk.Button(frame, text="Start Linker Training", command=self.run_link_train)
        self.link_start_btn.grid(row=row, column=0, pady=20)
        self.link_stop_btn = ttk.Button(frame, text="Stop Execution", command=self.stop_execution, state=tk.DISABLED)
        self.link_stop_btn.grid(row=row, column=1, pady=20)

    def run_link_train(self):
        ann_json = self.link_ann_entry.get()
        epochs = int(self.link_epochs_entry.get())
        mname = self.link_model_entry.get().strip()
        log_file = self.link_train_log_entry.get().strip() or None
        use_gnn = self.link_use_gnn_var.get()

        if not ann_json:
            messagebox.showwarning("Input Required", "Please provide link annotations.")
            return

        self.run_wrapper(self.link_start_btn, self.link_stop_btn, train_linker, ann_json, epochs=epochs, model_name=mname, log_file=log_file, use_gnn=use_gnn)

    # --- TAB: Association ---
    def create_association_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="4. Track Tips")

        row = 0
        ttk.Label(frame, text="Features JSON:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.assoc_feat_entry = ttk.Entry(frame, width=50)
        self.assoc_feat_entry.insert(0, "output/deep_tip_features.json")
        self.assoc_feat_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.assoc_feat_entry)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Linker Model Checkpoint:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.assoc_model_entry = ttk.Entry(frame, width=50)
        self.assoc_model_entry.insert(0, "models/gnn_linker.pth")
        self.assoc_model_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.assoc_model_entry)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Output Tracks JSON:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.assoc_json_entry = ttk.Entry(frame, width=50)
        self.assoc_json_entry.insert(0, "output/tracks.json")
        self.assoc_json_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.assoc_json_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Positive Links JSON:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.assoc_links_json_entry = ttk.Entry(frame, width=50)
        self.assoc_links_json_entry.insert(0, "output/positive_links.json")
        self.assoc_links_json_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.assoc_links_json_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Spatial Threshold (px):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.assoc_spatial_entry = ttk.Entry(frame, width=10)
        self.assoc_spatial_entry.insert(0, "150")
        self.assoc_spatial_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Maximum distance (in pixels) for a tip to be associated between frames.")

        row += 1
        ttk.Label(frame, text="Probability Threshold:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.assoc_prob_entry = ttk.Entry(frame, width=10)
        self.assoc_prob_entry.insert(0, "0.2")
        self.assoc_prob_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Minimum linker confidence required to form a track.")

        row += 1
        ttk.Label(frame, text="Log File (optional):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.assoc_log_entry = ttk.Entry(frame, width=50)
        self.assoc_log_entry.insert(0, "output/logs/tracking.json")
        self.assoc_log_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.assoc_log_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        self.assoc_start_btn = ttk.Button(frame, text="Run Tip Tracking", command=self.run_association)
        self.assoc_start_btn.grid(row=row, column=0, pady=20)
        self.assoc_stop_btn = ttk.Button(frame, text="Stop Execution", command=self.stop_execution, state=tk.DISABLED)
        self.assoc_stop_btn.grid(row=row, column=1, pady=20)

    def run_association(self):
        feat_json = self.assoc_feat_entry.get()
        model_path = self.assoc_model_entry.get()
        out_json = self.assoc_json_entry.get()
        out_links_json = self.assoc_links_json_entry.get()
        spatial = int(self.assoc_spatial_entry.get())
        prob = float(self.assoc_prob_entry.get())
        log_file = self.assoc_log_entry.get().strip() or None

        if not feat_json or not model_path:
            messagebox.showwarning("Input Required", "Please provide features and linker model.")
            return

        self.run_wrapper(self.assoc_start_btn, self.assoc_stop_btn, associate_tips_multi_plant, feat_json, model_path, out_json, output_links_json=out_links_json, spatial_threshold=spatial, prob_threshold=prob, log_file=log_file)

    # --- TAB: Growth Speed ---
    def create_growth_speed_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="5. Growth Speed")

        row = 0
        ttk.Label(frame, text="Tracks JSON:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.speed_tracks_entry = ttk.Entry(frame, width=50)
        self.speed_tracks_entry.insert(0, "output/tracks.json")
        self.speed_tracks_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.speed_tracks_entry)).grid(row=row, column=2, padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Output from Step 4 (Track Tips). Contains tip positions over time.")

        row += 1
        ttk.Label(frame, text="Images Folder:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.speed_img_entry = ttk.Entry(frame, width=50)
        self.speed_img_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_dir(self.speed_img_entry)).grid(row=row, column=2, padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Same images folder used for detection. Needed to compute geodesic root lengths.")

        row += 1
        ttk.Label(frame, text="Output CSV:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.speed_csv_entry = ttk.Entry(frame, width=50)
        self.speed_csv_entry.insert(0, "output/growth_speeds.csv")
        self.speed_csv_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.speed_csv_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Downscale Factor:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.speed_downscale_entry = ttk.Entry(frame, width=10)
        self.speed_downscale_entry.insert(0, "0.25")
        self.speed_downscale_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        self.add_info_icon(frame, row, 3, "Downscale image crops before computing geodesic paths. Lower = faster but less precise.")

        row += 1
        self.speed_frangi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Use Frangi vesselness filter", variable=self.speed_frangi_var).grid(row=row, column=0, columnspan=2, sticky='w', padx=5)
        self.add_info_icon(frame, row, 3, "Frangi filter enhances root structures for better path accuracy. Uncheck for faster but less precise results.")

        row += 1
        self.speed_start_btn = ttk.Button(frame, text="Compute Growth Speeds", command=self.run_growth_speed)
        self.speed_start_btn.grid(row=row, column=0, pady=20)
        self.speed_stop_btn = ttk.Button(frame, text="Stop Execution", command=self.stop_execution, state=tk.DISABLED)
        self.speed_stop_btn.grid(row=row, column=1, pady=20)

    def run_growth_speed(self):
        tracks_file = self.speed_tracks_entry.get()
        img_folder = self.speed_img_entry.get()
        output_csv = self.speed_csv_entry.get()
        downscale = float(self.speed_downscale_entry.get())
        use_frangi = self.speed_frangi_var.get()

        if not tracks_file or not img_folder:
            messagebox.showwarning("Input Required", "Please provide tracks JSON and images folder.")
            return

        self.run_wrapper(self.speed_start_btn, self.speed_stop_btn, compute_incremental_speeds, tracks_file, img_folder, output_csv, downscale=downscale, use_frangi=use_frangi)

if __name__ == "__main__":
    root = tk.Tk()
    app = RootCNN_V2_GUI(root)
    root.mainloop()
