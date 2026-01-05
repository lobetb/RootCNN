# RootCNN

RootCNN is a deep learning pipeline for detecting and tracking root tips in time-series images. It provides a unified graphical interface to handle the entire workflow from training to association.

## Key Features

- **Unified GUI**: A single, easy-to-use interface for all steps of the pipeline.
- **High Performance**: Fully vectorized GPU-accelerated tracking and optimized patch-based inference.
- **Configurable Thresholds**: Set custom score thresholds for both tip detection and link association.
- **Artifact Filtering**: Pre-analysis by a small CNN detects noisy images (e.g. aeroponics photos that have drops in the image that can be detected as tips).

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
python gui.py
```

### The Workflow

The GUI is organized into four sequential tabs:

0. **Noisy images detection** If you have random noisy images, you can train a model to recognize them and eliminate them automatically to avoid a lot of computing overhead down the line
1. **Train Detector**: Train the UNet model to identify root tips based on annotated ground truth.
2. **Detect Tips**: Use a trained detector to predict tips and optionally extract deep features from your image series. Features are necessary for the subsequent tracking step.

The two remaining functions don't work as they should right now, the idea is to train a MLP affinity model with ground-truth associations of tips between an image at t time and the image of the same plant at t+1, based on coordinates and the 256-dimensions vector of deep features extracted during the tips detection.

## Project Structure

- `gui.py`: Main entry point for the graphical interface.
- `src/`: Core source code.
  - `detection/`: UNet models and tip detection logic.
  - `association/`: Affinity MLP models and tip tracking logic.
  - `utils/`: Shared utilities (path handling, device management, outlier filtering).
- `models/`: Default directory for trained model checkpoints (`.pth`).
- `output/`: Default directory for exported features and tracking results (`.json`).

## Performance & Benchmarking

RootCNN is optimized for high throughput using PyTorch vectorization. Most of the heavy computation (including tiling, inference, and association) happens directly on the GPU.

Performance on a home PC with a Ryzen 7 5800X3D processor, 64 go ram and a Geforce RTX 4080 GPU on 14k*4k images from the aeroponics platform of UCLouvain ( [rootphair](https://www.uclouvain.be/en/research-institutes/eli/elia/rootphair) ) :

- Training the detection model on three annotated images (>1000 tips annotated) takes around 25 minutes for 20 epochs and 4 workers. It is possible to run with 8 workers on 16 go VRAM.
<img width="1000" height="268" alt="Training time per epoch" src="https://github.com/user-attachments/assets/d66ef545-f547-4d4d-a4ba-d2f15b47dfef" />

- 20 epochs seem sufficient with only three images to get to a reliable detection :
<img width="1000" height="238" alt="Training and validation loss" src="https://github.com/user-attachments/assets/c36731b9-a35d-4872-a287-e710a84cb1a3" />

- Detection of the tips and extraction of deep features takes between 7 and 20 secondes per image, depending on the number of tips :
<img width="1000" height="280" alt="Computing time by tip count" src="https://github.com/user-attachments/assets/2b841a07-988b-43c6-bb70-fe409996c684" />

- There seems to be some false positive on some images that have more tips detected, it is under investigation :
<img width="1000" height="277" alt="Screenshot from 2026-01-05 15-43-15" src="https://github.com/user-attachments/assets/d4cb60e1-6059-431a-95ac-bb285c1a3413" />


## License
Under MIT License
