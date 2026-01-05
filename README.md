# RootCNN v2

RootCNN v2 is a consolidated deep learning pipeline for detecting and tracking root tips in time-series images. It provides a unified graphical interface to handle the entire workflow from training to association.

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
python RootCNN_v2/gui.py
```

### The Workflow

The GUI is organized into four sequential tabs:

1. **Train Detector**: Train the UNet model to identify root tips based on annotated ground truth.
2. **Detect Tips**: Use a trained detector to predict tips and optionally extract deep features from your image series. Features are necessary for the subsequent tracking step.
3. **Train Linker**: Train the Affinity MLP to learn the association between tips in consecutive frames using deep features and spatial relative coordinates.
4. **Track Tips**: Perform the final association and tracking across the series. This step automatically handles multiple plant sequences and filters out images containing artifacts.

## Project Structure

- `gui.py`: Main entry point for the graphical interface.
- `src/`: Core source code.
  - `detection/`: UNet models and tip detection logic.
  - `association/`: Affinity MLP models and tip tracking logic.
  - `utils/`: Shared utilities (path handling, device management, outlier filtering).
- `models/`: Default directory for trained model checkpoints (`.pth`).
- `output/`: Default directory for exported features and tracking results (`.json`).
- `tests/`: Benchmarking and verification scripts.

## Performance & Benchmarking

RootCNN v2 is optimized for high throughput using PyTorch vectorization. Most of the heavy computation (including tiling, inference, and association) happens directly on the GPU.

To run the performance benchmark:
```bash
./env/bin/python3 tests/benchmark_optimization.py
```

Typical performance on a modern GPU:
- **Detection**: ~1.1s for a 2048x2048 image.
- **Tracking**: ~0.08s for 100x100 tip associations (10,000 pairs).

## License

[Add License Information Here]
