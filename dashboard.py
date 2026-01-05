"""
RootCNN v2 Performance Dashboard
Visualizes training and inference metrics from log files using Plotly/Dash.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Initialize Dash app
app = dash.Dash(__name__, title="RootCNN Dashboard")

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("RootCNN Performance Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
        
        # File selection section
        html.Div([
            html.H3("Select Log Files", style={'color': '#34495e'}),
            html.Div([
                html.Label("Detection Log:", style={'fontWeight': 'bold'}),
                dcc.Input(id='detection-log-path', type='text', 
                         value='output/logs/detection.json',
                         style={'width': '100%', 'padding': '8px', 'marginBottom': '10px'}),
                
                html.Label("Tracking Log:", style={'fontWeight': 'bold'}),
                dcc.Input(id='tracking-log-path', type='text', 
                         value='output/logs/tracking.json',
                         style={'width': '100%', 'padding': '8px', 'marginBottom': '10px'}),
                
                html.Label("Training Log (Detection):", style={'fontWeight': 'bold'}),
                dcc.Input(id='training-log-path', type='text', 
                         value='output/logs/detection_training.json',
                         style={'width': '100%', 'padding': '8px', 'marginBottom': '10px'}),
                
                html.Button('Load Data', id='load-button', n_clicks=0,
                           style={'padding': '10px 20px', 'backgroundColor': '#3498db', 
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px',
                                  'cursor': 'pointer', 'fontSize': '16px'}),
                
                html.Div(id='load-status', style={'marginTop': '10px', 'color': '#27ae60'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px'})
        ], style={'marginBottom': '30px'}),
        
        # Tabs for different visualizations
        dcc.Tabs(id='tabs', value='tab-detection', children=[
            dcc.Tab(label='Detection Metrics', value='tab-detection', 
                   style={'padding': '10px', 'fontWeight': 'bold'}),
            dcc.Tab(label='Tracking Metrics', value='tab-tracking',
                   style={'padding': '10px', 'fontWeight': 'bold'}),
            dcc.Tab(label='Training Metrics', value='tab-training',
                   style={'padding': '10px', 'fontWeight': 'bold'}),
            dcc.Tab(label='Performance Analysis', value='tab-performance',
                   style={'padding': '10px', 'fontWeight': 'bold'}),
        ]),
        
        html.Div(id='tabs-content', style={'marginTop': '20px'})
    ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'})
])

def load_log_file(filepath):
    """Load a log file and return the data."""
    try:
        path = Path(filepath)
        if not path.exists():
            return None, f"File not found: {filepath}"
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data, None
    except Exception as e:
        return None, f"Error loading {filepath}: {str(e)}"

def create_detection_plots(data):
    """Create plots for detection metrics."""
    if not data or 'entries' not in data:
        return html.Div("No detection data available", style={'color': 'red'})
    
    df = pd.DataFrame(data['entries'])
    
    # Convert timestamps
    df['image_timestamp'] = pd.to_datetime(df['image_timestamp'])
    df = df.sort_values('image_timestamp')
    
    plots = []
    
    # Plot 1: Tips detected over time
    fig1 = px.line(df, x='image_timestamp', y='num_tips',
                   title='Root Tips Detected Over Time',
                   labels={'image_timestamp': 'Time', 'num_tips': 'Number of Tips'})
    fig1.update_traces(line=dict(color='#2ecc71', width=3))
    fig1.update_layout(hovermode='x unified', height=400)
    plots.append(dcc.Graph(figure=fig1))
    
    # Plot 2: Processing time over time
    fig2 = px.line(df, x='image_timestamp', y='processing_time_seconds',
                   title='Processing Time Over Time',
                   labels={'image_timestamp': 'Time', 'processing_time_seconds': 'Processing Time (s)'})
    fig2.update_traces(line=dict(color='#e74c3c', width=3))
    fig2.update_layout(hovermode='x unified', height=400)
    plots.append(dcc.Graph(figure=fig2))
    
    # Plot 3: Processing time vs number of tips (scatter)
    fig3 = px.scatter(df, x='num_tips', y='processing_time_seconds',
                      title='Processing Time vs. Number of Tips',
                      labels={'num_tips': 'Number of Tips', 
                             'processing_time_seconds': 'Processing Time (s)'},
                      trendline='ols')
    fig3.update_traces(marker=dict(size=10, color='#9b59b6'))
    fig3.update_layout(height=400)
    plots.append(dcc.Graph(figure=fig3))
    
    # Summary statistics
    stats = html.Div([
        html.H4("Summary Statistics", style={'color': '#2c3e50'}),
        html.Div([
            html.Div([
                html.P(f"Total Images: {len(df)}", style={'fontSize': '18px'}),
                html.P(f"Avg Tips/Image: {df['num_tips'].mean():.1f}", style={'fontSize': '18px'}),
                html.P(f"Max Tips: {df['num_tips'].max()}", style={'fontSize': '18px'}),
            ], style={'display': 'inline-block', 'marginRight': '50px'}),
            html.Div([
                html.P(f"Avg Processing Time: {df['processing_time_seconds'].mean():.2f}s", 
                      style={'fontSize': '18px'}),
                html.P(f"Total Processing Time: {df['processing_time_seconds'].sum():.2f}s", 
                      style={'fontSize': '18px'}),
                html.P(f"Time Range: {(df['image_timestamp'].max() - df['image_timestamp'].min()).days} days", 
                      style={'fontSize': '18px'}),
            ], style={'display': 'inline-block'}),
        ])
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'marginBottom': '20px'})
    
    return html.Div([stats] + plots)

def create_tracking_plots(data):
    """Create plots for tracking metrics."""
    if not data or 'entries' not in data:
        return html.Div("No tracking data available", style={'color': 'red'})
    
    df = pd.DataFrame(data['entries'])
    
    # Convert timestamps
    df['image_timestamp'] = pd.to_datetime(df['image_timestamp'])
    df = df.sort_values('image_timestamp')
    
    plots = []
    
    # Plot 1: Tips tracked over time by plant
    if 'plant_id' in df.columns:
        fig1 = px.line(df, x='image_timestamp', y='num_tips', color='plant_id',
                       title='Root Tips Tracked Over Time (by Plant)',
                       labels={'image_timestamp': 'Time', 'num_tips': 'Number of Tips'})
        fig1.update_layout(hovermode='x unified', height=400)
        plots.append(dcc.Graph(figure=fig1))
    else:
        fig1 = px.line(df, x='image_timestamp', y='num_tips',
                       title='Root Tips Tracked Over Time',
                       labels={'image_timestamp': 'Time', 'num_tips': 'Number of Tips'})
        fig1.update_traces(line=dict(color='#2ecc71', width=3))
        fig1.update_layout(hovermode='x unified', height=400)
        plots.append(dcc.Graph(figure=fig1))
    
    # Plot 2: Association success rate
    if 'num_associations' in df.columns:
        df['association_rate'] = (df['num_associations'] / df['num_tips'] * 100).fillna(0)
        
        fig2 = px.line(df, x='image_timestamp', y='association_rate',
                       color='plant_id' if 'plant_id' in df.columns else None,
                       title='Association Success Rate Over Time',
                       labels={'image_timestamp': 'Time', 'association_rate': 'Association Rate (%)'})
        fig2.update_layout(hovermode='x unified', height=400)
        plots.append(dcc.Graph(figure=fig2))
        
        # Plot 3: Associations vs Tips
        fig3 = px.scatter(df, x='num_tips', y='num_associations',
                         color='plant_id' if 'plant_id' in df.columns else None,
                         title='Successful Associations vs. Total Tips',
                         labels={'num_tips': 'Total Tips', 'num_associations': 'Successful Associations'},
                         trendline='ols')
        fig3.update_layout(height=400)
        plots.append(dcc.Graph(figure=fig3))
    
    # Summary statistics
    stats_content = [
        html.P(f"Total Images: {len(df)}", style={'fontSize': '18px'}),
        html.P(f"Avg Tips/Image: {df['num_tips'].mean():.1f}", style={'fontSize': '18px'}),
    ]
    
    if 'num_associations' in df.columns:
        avg_assoc_rate = (df['num_associations'].sum() / df['num_tips'].sum() * 100)
        stats_content.append(
            html.P(f"Avg Association Rate: {avg_assoc_rate:.1f}%", style={'fontSize': '18px'})
        )
    
    if 'plant_id' in df.columns:
        stats_content.append(
            html.P(f"Number of Plants: {df['plant_id'].nunique()}", style={'fontSize': '18px'})
        )
    
    stats = html.Div([
        html.H4("Summary Statistics", style={'color': '#2c3e50'}),
        html.Div(stats_content)
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'marginBottom': '20px'})
    
    return html.Div([stats] + plots)

def create_training_plots(data):
    """Create plots for training metrics."""
    if not data or 'entries' not in data:
        return html.Div("No training data available", style={'color': 'red'})
    
    df = pd.DataFrame(data['entries'])
    
    plots = []
    
    # Plot 1: Loss over epochs
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['epoch'], y=df['train_loss'], 
                              mode='lines+markers', name='Train Loss',
                              line=dict(color='#3498db', width=3)))
    fig1.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], 
                              mode='lines+markers', name='Val Loss',
                              line=dict(color='#e74c3c', width=3)))
    fig1.update_layout(title='Training and Validation Loss',
                      xaxis_title='Epoch', yaxis_title='Loss',
                      hovermode='x unified', height=400)
    plots.append(dcc.Graph(figure=fig1))
    
    # Plot 2: Accuracy over epochs (if available)
    if 'train_accuracy' in df.columns and 'val_accuracy' in df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['epoch'], y=df['train_accuracy'], 
                                  mode='lines+markers', name='Train Accuracy',
                                  line=dict(color='#2ecc71', width=3)))
        fig2.add_trace(go.Scatter(x=df['epoch'], y=df['val_accuracy'], 
                                  mode='lines+markers', name='Val Accuracy',
                                  line=dict(color='#f39c12', width=3)))
        fig2.update_layout(title='Training and Validation Accuracy',
                          xaxis_title='Epoch', yaxis_title='Accuracy',
                          hovermode='x unified', height=400)
        plots.append(dcc.Graph(figure=fig2))
    
    # Plot 3: Epoch time
    if 'epoch_time_seconds' in df.columns:
        fig3 = px.bar(df, x='epoch', y='epoch_time_seconds',
                      title='Training Time per Epoch',
                      labels={'epoch': 'Epoch', 'epoch_time_seconds': 'Time (seconds)'})
        fig3.update_traces(marker_color='#9b59b6')
        fig3.update_layout(height=400)
        plots.append(dcc.Graph(figure=fig3))
    
    # Summary statistics
    stats = html.Div([
        html.H4("Training Summary", style={'color': '#2c3e50'}),
        html.Div([
            html.P(f"Total Epochs: {len(df)}", style={'fontSize': '18px'}),
            html.P(f"Final Train Loss: {df['train_loss'].iloc[-1]:.4f}", style={'fontSize': '18px'}),
            html.P(f"Final Val Loss: {df['val_loss'].iloc[-1]:.4f}", style={'fontSize': '18px'}),
            html.P(f"Best Val Loss: {df['val_loss'].min():.4f} (Epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})", 
                  style={'fontSize': '18px'}),
        ])
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'marginBottom': '20px'})
    
    return html.Div([stats] + plots)

def create_performance_analysis(detection_data, tracking_data):
    """Create comparative performance analysis."""
    plots = []
    
    # Compare detection vs tracking processing times
    if detection_data and tracking_data:
        det_df = pd.DataFrame(detection_data['entries'])
        track_df = pd.DataFrame(tracking_data['entries'])
        
        # Processing time comparison
        fig1 = go.Figure()
        fig1.add_trace(go.Box(y=det_df['processing_time_seconds'], name='Detection',
                             marker_color='#3498db'))
        fig1.add_trace(go.Box(y=track_df['processing_time_seconds'], name='Tracking',
                             marker_color='#2ecc71'))
        fig1.update_layout(title='Processing Time Distribution: Detection vs Tracking',
                          yaxis_title='Processing Time (seconds)', height=400)
        plots.append(dcc.Graph(figure=fig1))
        
        # Tips comparison
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=det_df['num_tips'], name='Detection',
                             marker_color='#e74c3c'))
        fig2.add_trace(go.Box(y=track_df['num_tips'], name='Tracking',
                             marker_color='#f39c12'))
        fig2.update_layout(title='Number of Tips Distribution: Detection vs Tracking',
                          yaxis_title='Number of Tips', height=400)
        plots.append(dcc.Graph(figure=fig2))
    
    if not plots:
        return html.Div("Load both detection and tracking logs for performance analysis", 
                       style={'color': '#7f8c8d', 'fontSize': '18px', 'textAlign': 'center', 'padding': '50px'})
    
    return html.Div(plots)

# Callbacks
@app.callback(
    [Output('tabs-content', 'children'),
     Output('load-status', 'children')],
    [Input('load-button', 'n_clicks'),
     Input('tabs', 'value')],
    [State('detection-log-path', 'value'),
     State('tracking-log-path', 'value'),
     State('training-log-path', 'value')]
)
def update_content(n_clicks, tab, detection_path, tracking_path, training_path):
    """Update dashboard content based on selected tab and loaded data."""
    
    # Load data
    detection_data, det_error = load_log_file(detection_path)
    tracking_data, track_error = load_log_file(tracking_path)
    training_data, train_error = load_log_file(training_path)
    
    # Status message
    status_msgs = []
    if det_error:
        status_msgs.append(f"⚠️ {det_error}")
    else:
        status_msgs.append(f"✓ Detection log loaded ({detection_data.get('num_entries', 0)} entries)")
    
    if track_error:
        status_msgs.append(f"⚠️ {track_error}")
    else:
        status_msgs.append(f"✓ Tracking log loaded ({tracking_data.get('num_entries', 0)} entries)")
    
    if train_error:
        status_msgs.append(f"⚠️ {train_error}")
    else:
        status_msgs.append(f"✓ Training log loaded ({training_data.get('num_entries', 0)} entries)")
    
    status = html.Div([html.P(msg, style={'margin': '5px 0'}) for msg in status_msgs])
    
    # Render appropriate tab content
    if tab == 'tab-detection':
        content = create_detection_plots(detection_data)
    elif tab == 'tab-tracking':
        content = create_tracking_plots(tracking_data)
    elif tab == 'tab-training':
        content = create_training_plots(training_data)
    elif tab == 'tab-performance':
        content = create_performance_analysis(detection_data, tracking_data)
    else:
        content = html.Div("Select a tab to view metrics")
    
    return content, status

if __name__ == '__main__':
    print("=" * 60)
    print("RootCNN v2 Performance Dashboard")
    print("=" * 60)
    print("\nStarting dashboard server...")
    print("Open your browser and navigate to: http://127.0.0.1:8050")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=8050)
