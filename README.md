# RootCNN

RootCNN is a consolidated deep learning pipeline for detecting and tracking root tips in time-series images. It provides a unified graphical interface to handle the entire workflow from training to association.

## Key Features

- **Unified GUI**: A single, easy-to-use interface for all steps of the pipeline.
- **High Performance**: Fully vectorized GPU-accelerated tracking and optimized patch-based inference.
- **Support for Multiple Plants**: Automatically groups images by plant ID (extracted from filenames, e.g., `A_03_5`) and tracks them independently.
- **Recursive Image Discovery**: Scans folders and subfolders for images, allowing for better data organization.
- **Configurable Thresholds**: Set custom score thresholds for both tip detection and link association.
- **Artifact Filtering**: Automatically detects and skips images with suspicious tip counts (outliers) to ensure tracking robustness.
- **Positive Links Export**: Option to export a dedicated file containing only successful tip-to-tip associations with their pixel distances.
- **Clean Architecture**: Reorganized codebase for modularity and maintainability.

## Installation

1. Clone the repository.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Main dependencies include:
- PyTorch
- OpenCV
- NumPy
- scikit-image
- tqdm
- Pillow
- Matplotlib

## Usage

Launch the main interface by running:
```bash
python RootCNN/gui.py
```

### The Workflow

The GUI is organized into four sequential tabs:

0. **Train Noisy Images Detector**: Train to recognize noisy images (where droplets from the water supply can induce false positives).
1. **Train Detector**: Train the UNet model to identify root tips based on annotated ground truth.
2. **Detect Tips**: Use a trained detector to predict tips and optionally extract deep features from your image series. Features are necessary for the subsequent tracking step.
3. **Train Linker**: Train the GNN (graphe neural network) to learn the association between tips in consecutive frames using deep features and spatial relative coordinates.
4. **Track Tips**: Perform the final association and tracking across the series. This step automatically handles multiple plant sequences and filters out images containing artifacts.

## Project Structure

- `gui.py`: Main entry point for the graphical interface.
- `src/`: Core source code.
  - `detection/`: UNet models and tip detection logic.
  - `association/`: Affinity MLP models and tip tracking logic.
  - `utils/`: Shared utilities (path handling, device management, outlier filtering).
- `models/`: Default directory for trained model checkpoints (`.pth`).
- `output/`: Default directory for exported features and tracking results (`.json`). Created when running the program.
- `tools/`: Three tools.
  - `generate_heatmap.py`: Generate images to visualize tips detection.
  - `link_annotator.py`: Manually annotate the links between tips in image pairs (requires tips annotation first).
  - `tips_annotator.py`: Manually annotate tips in images.

## Performance & Benchmarking

RootCNN is optimized for high throughput using PyTorch vectorization. Most of the heavy computation (including tiling, inference, and association) happens directly on the GPU. Mixed precision (FP16) is used for performance.

Current performance on a RTX 4080 GPU:
- **Detection**: ~7s per 14k*4k from the aeroponics platform
- **Tracking**: Very fast, not measured yet.

Preliminary evaluation of the linking accuracy (on a model trained on a limited dataset of 16 image pairs) shows that it identifies around 50% of the ground-truth links and that 70% of the identified links are correct.
