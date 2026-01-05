# RootCNN v2 Performance Dashboard

## Overview

Interactive web-based dashboard for visualizing RootCNN v2 performance metrics using Plotly and Dash.

## Features

### 📊 Detection Metrics Tab
- **Tips Detected Over Time**: Line chart showing root growth progression
- **Processing Time Over Time**: Monitor performance across your image series
- **Processing Time vs. Tips**: Scatter plot with trendline to analyze computational complexity
- **Summary Statistics**: Quick overview of detection performance

### 🔗 Tracking Metrics Tab
- **Tips Tracked Over Time**: Multi-plant visualization of root tip tracking
- **Association Success Rate**: Monitor how well tips are linked between frames
- **Associations vs. Tips**: Correlation analysis
- **Summary Statistics**: Tracking performance overview

### 📈 Training Metrics Tab
- **Loss Curves**: Training and validation loss over epochs
- **Accuracy Curves**: Training and validation accuracy progression
- **Epoch Time**: Training duration analysis
- **Training Summary**: Final metrics and best epoch

### ⚡ Performance Analysis Tab
- **Detection vs. Tracking**: Comparative box plots for processing times
- **Tips Distribution**: Compare detection and tracking tip counts

## Installation

The required packages are already installed if you ran the pip install command. If not:

```bash
env/bin/pip install plotly dash pandas
```

## Usage

### 1. Generate Log Files

First, run your RootCNN workflows with logging enabled. You can use the GUI or Python directly:

**Via GUI:**
1. Open `gui.py`
2. Fill in the log file paths in each tab (or use defaults)
3. Run your workflows

**Via Python:**
```python
from src.detection.core import export_features_for_folder

export_features_for_folder(
    img_folder="data/images",
    model_ckpt="models/detector.pth",
    output_json="output/features.json",
    log_file="output/logs/my_detection.json"  # Enable logging
)
```

### 2. Start the Dashboard

```bash
env/bin/python dashboard.py
```

The dashboard will start on `http://127.0.0.1:8050`

### 3. Load Your Data

1. Open your browser to `http://127.0.0.1:8050`
2. Enter the paths to your log files in the input fields:
   - Detection Log: `output/logs/detection.json`
   - Tracking Log: `output/logs/tracking.json`
   - Training Log: `output/logs/detection_training.json`
3. Click "Load Data"
4. Navigate between tabs to explore different metrics

## Sample Data

To test the dashboard with sample data:

```bash
# Generate sample logs
env/bin/python example_logging.py

# Start dashboard (it's already configured to use sample logs)
env/bin/python dashboard.py
```

## Dashboard Features

### Interactive Plots
- **Zoom**: Click and drag to zoom into specific regions
- **Pan**: Hold shift and drag to pan
- **Hover**: Hover over data points for detailed information
- **Reset**: Double-click to reset view

### Real-time Updates
- Change log file paths and click "Load Data" to refresh
- Switch between tabs to view different metrics
- All plots update automatically

## Customization

You can customize the dashboard by editing `dashboard.py`:

- **Colors**: Modify the `line=dict(color='#hexcode')` in plot definitions
- **Layout**: Adjust plot heights with `height=400` parameter
- **Metrics**: Add new plots by creating functions similar to `create_detection_plots()`
- **Styling**: Update CSS in the `style={}` dictionaries

## Troubleshooting

**Dashboard won't start:**
- Make sure port 8050 is not in use
- Try a different port: `app.run(debug=True, port=8051)`

**Log files not loading:**
- Check that the file paths are correct
- Ensure log files are valid JSON
- Look at the status messages below the "Load Data" button

**Plots not showing:**
- Verify your log files have data entries
- Check the browser console for JavaScript errors
- Try refreshing the page

**Performance issues:**
- For large datasets (>1000 entries), consider sampling your data
- Disable debug mode: `app.run(debug=False)`

## Tips for Best Results

1. **Consistent Logging**: Always use the same log file paths for similar experiments
2. **Descriptive Names**: Use meaningful log file names (e.g., `plant1_week1_detection.json`)
3. **Regular Monitoring**: Check the dashboard during long training runs
4. **Compare Experiments**: Load different log files to compare performance

## Advanced Usage

### Running on a Remote Server

If you're running RootCNN on a remote server:

```bash
# On server
env/bin/python dashboard.py

# On local machine, create SSH tunnel
ssh -L 8050:localhost:8050 user@server
```

Then access `http://localhost:8050` on your local machine.

### Exporting Plots

To save plots as images:
1. Hover over any plot
2. Click the camera icon in the top-right
3. The plot will download as a PNG

### Batch Processing

To generate reports for multiple experiments:

```python
# Create a script to load and compare multiple log files
import json
import pandas as pd

logs = [
    "output/logs/experiment1_detection.json",
    "output/logs/experiment2_detection.json",
]

for log_file in logs:
    with open(log_file) as f:
        data = json.load(f)
    df = pd.DataFrame(data['entries'])
    print(f"\n{log_file}:")
    print(f"  Avg tips: {df['num_tips'].mean():.1f}")
    print(f"  Avg time: {df['processing_time_seconds'].mean():.2f}s")
```

## Support

For issues or questions:
1. Check the walkthrough documentation
2. Review the example_logging.py script
3. Inspect sample log files for correct format
