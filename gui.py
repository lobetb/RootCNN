import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import sys
import os
import queue
from pathlib import Path

# Fix python path to allow imports from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.core import train_detection, export_features_for_folder
from src.association.core import train_linker, associate_tips_multi_plant
from src.detection.noise import train_noise_classifier # Import from src


class ThreadSafeConsole:
    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root
        self.queue = queue.Queue()
        self.update_interval = 100
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

class RootCNN_V2_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RootCNN v2 - Consolidated Interface")
        self.root.geometry("900x700")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tabs
        # Tabs
        self.create_noise_train_tab() # New tab
        self.create_detection_train_tab()
        self.create_detect_tips_tab()
        self.create_linker_train_tab()
        self.create_association_tab()

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

    def run_wrapper(self, func, *args, **kwargs):
        def task():
            try:
                func(*args, **kwargs)
                messagebox.showinfo("Success", "Operation completed successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
        threading.Thread(target=task, daemon=True).start()

    # --- TAB: Noise Training ---
    def create_noise_train_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="0. Train Noise Filter")
        
        row = 0
        ttk.Label(frame, text="Dataset Folder (contains 'clean'/'noisy'):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.noise_train_data_entry = ttk.Entry(frame, width=50)
        self.noise_train_data_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_dir(self.noise_train_data_entry)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Epochs:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.noise_epochs_entry = ttk.Entry(frame, width=10)
        self.noise_epochs_entry.insert(0, "15")
        self.noise_epochs_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Output Model Name:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.noise_model_name_entry = ttk.Entry(frame, width=30)
        self.noise_model_name_entry.insert(0, "models/noise_classifier.pth")
        self.noise_model_name_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)

        row += 1
        ttk.Button(frame, text="Start Noise Model Training", command=self.run_noise_train).grid(row=row, column=0, columnspan=3, pady=20)

    def run_noise_train(self):
        data_dir = self.noise_train_data_entry.get()
        epochs = int(self.noise_epochs_entry.get())
        output_model = self.noise_model_name_entry.get()

        if not data_dir:
            messagebox.showwarning("Input Required", "Please provide dataset folder.")
            return
            
        # Create output directory for model if needed
        os.makedirs(os.path.dirname(output_model), exist_ok=True)

        self.run_wrapper(train_noise_classifier, data_dir, output_model, epochs=epochs)

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

        row += 1
        ttk.Label(frame, text="Batch Size:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.det_batch_entry = ttk.Entry(frame, width=10)
        self.det_batch_entry.insert(0, "4")
        self.det_batch_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)

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
        ttk.Button(frame, text="Start Detector Training", command=self.run_det_train).grid(row=row, column=0, columnspan=3, pady=20)

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

        self.run_wrapper(train_detection, img_dir, ann_file, epochs=epochs, batch_size=bs, model_name=mname, log_file=log_file)

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

        row += 1
        self.exp_extract_feat_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Export deep features (necessary for tracking)", variable=self.exp_extract_feat_var).grid(row=row, column=0, columnspan=2, sticky='w', padx=5)

        row += 1
        self.exp_use_gt_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Use Ground Truth Tips?", variable=self.exp_use_gt_var).grid(row=row, column=0, columnspan=2, sticky='w', padx=5)

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

        row += 1
        ttk.Label(frame, text="Detection log file:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.exp_log_entry = ttk.Entry(frame, width=50)
        self.exp_log_entry.insert(0, "output/logs/detection.json")
        self.exp_log_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.exp_log_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)
        
        row += 1
        self.exp_filter_noise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Filter Noisy Images?", variable=self.exp_filter_noise_var).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        
        row += 1
        ttk.Label(frame, text="Noise Model Path:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.exp_noise_model_entry = ttk.Entry(frame, width=50)
        self.exp_noise_model_entry.insert(0, "models/noise_classifier.pth")
        self.exp_noise_model_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.exp_noise_model_entry)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Button(frame, text="Run Detection & Feature Extraction", command=self.run_feature_export).grid(row=row, column=0, columnspan=3, pady=20)

    def run_feature_export(self):
        img_folder = self.exp_img_entry.get()
        model_ckpt = self.exp_model_entry.get()
        out_json = self.exp_json_entry.get()
        use_gt = self.exp_use_gt_var.get()
        extract_feat = self.exp_extract_feat_var.get()
        thresh = float(self.exp_thresh_entry.get())
        ann = self.exp_ann_entry.get()
        log_file = self.exp_log_entry.get().strip() or None
        
        filter_noise = self.exp_filter_noise_var.get()
        noise_model_path = self.exp_noise_model_entry.get()

        if not img_folder or not model_ckpt:
            messagebox.showwarning("Input Required", "Please provide image folder and model.")
            return

        self.run_wrapper(export_features_for_folder, img_folder, model_ckpt, out_json, annotations_json=ann, use_gt=use_gt, extract_features=extract_feat, threshold=thresh, log_file=log_file, filter_noise=filter_noise, noise_model_path=noise_model_path)

    # --- TAB: Linker Training ---
    def create_linker_train_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="3. Train Linker")

        row = 0
        ttk.Label(frame, text="Features JSON:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.link_feat_entry = ttk.Entry(frame, width=50)
        self.link_feat_entry.insert(0, "output/deep_tip_features.json")
        self.link_feat_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.link_feat_entry)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Link Annotations JSON:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.link_ann_entry = ttk.Entry(frame, width=50)
        self.link_ann_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.link_ann_entry)).grid(row=row, column=2, padx=5, pady=5)

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

        row += 1
        ttk.Label(frame, text="Log File (optional):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.link_train_log_entry = ttk.Entry(frame, width=50)
        self.link_train_log_entry.insert(0, "output/logs/linker_training.json")
        self.link_train_log_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.link_train_log_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Button(frame, text="Start Linker Training", command=self.run_link_train).grid(row=row, column=0, columnspan=3, pady=20)

    def run_link_train(self):
        feat_json = self.link_feat_entry.get()
        ann_json = self.link_ann_entry.get()
        epochs = int(self.link_epochs_entry.get())
        mname = self.link_model_entry.get().strip()
        log_file = self.link_train_log_entry.get().strip() or None
        use_gnn = self.link_use_gnn_var.get()

        if not feat_json or not ann_json:
            messagebox.showwarning("Input Required", "Please provide features and link annotations.")
            return

        self.run_wrapper(train_linker, feat_json, ann_json, epochs=epochs, model_name=mname, log_file=log_file, use_gnn=use_gnn)

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

        row += 1
        ttk.Label(frame, text="Probability Threshold:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.assoc_prob_entry = ttk.Entry(frame, width=10)
        self.assoc_prob_entry.insert(0, "0.2")
        self.assoc_prob_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Log File (optional):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.assoc_log_entry = ttk.Entry(frame, width=50)
        self.assoc_log_entry.insert(0, "output/logs/tracking.json")
        self.assoc_log_entry.grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.assoc_log_entry, save=True)).grid(row=row, column=2, padx=5, pady=5)

        row += 1
        ttk.Button(frame, text="Run Tip Tracking", command=self.run_association).grid(row=row, column=0, columnspan=3, pady=20)

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

        self.run_wrapper(associate_tips_multi_plant, feat_json, model_path, out_json, output_links_json=out_links_json, spatial_threshold=spatial, prob_threshold=prob, log_file=log_file)

if __name__ == "__main__":
    root = tk.Tk()
    app = RootCNN_V2_GUI(root)
    root.mainloop()
