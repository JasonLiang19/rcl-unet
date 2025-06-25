# seq = 'MKRRQKRKHLENEESQETAEKGGGMSKSQE' # rcl: EKGGGMS
# label_start = 20 # 1 indexed 
# label_end = 26
# predictions = [6,7,20,21,22,23,24,25,26,27] # 0 indexed 
# probabilities = [0.09, 0.09, 0.09, 0.09, 0.13, 0.27, 0.59, 0.60, 0.27, 0.13,
#                 0.10, 0.08, 0.07, 0.07, 0.06, 0.07, 0.07, 0.07, 0.09, 0.25,
#                 0.54, 0.77, 0.83, 0.80, 0.76, 0.75, 0.72, 0.56, 0.27, 0.10]

import plotly.graph_objects as go
import numpy as np

# Example data
sequence_length = 450
np.random.seed(42)
probs = np.random.beta(a=2, b=8, size=sequence_length) * 0.4
probs[[6, 7, 20, 21, 22, 23, 24, 25, 26, 27]] = np.random.uniform(0.6, 0.9, 10)
from scipy.ndimage import gaussian_filter1d
smoothed_probs = gaussian_filter1d(probs, sigma=1)

# Feature annotations
pfam_start, pfam_end = 50, 420
expdis_start, expdis_end = 20, 40
ptm_position = 440

# Create figure
fig = go.Figure()

# Line plot for probability
fig.add_trace(go.Scatter(
    x=list(range(sequence_length)),
    y=smoothed_probs,
    mode='lines',
    line=dict(color='red'),
    name='RCL probability'
))

# PFAM domain bar
fig.add_trace(go.Bar(
    x=[(pfam_start + pfam_end) / 2],
    y=['PFAM'],
    width=[pfam_end - pfam_start],
    marker_color='skyblue',
    orientation='h',
    name='PFAM'
))

# EXP DIS region bar
fig.add_trace(go.Bar(
    x=[(expdis_start + expdis_end) / 2],
    y=['EXP DIS'],
    width=[expdis_end - expdis_start],
    marker_color='firebrick',
    orientation='h',
    name='EXP DIS'
))

# PTM marker
fig.add_trace(go.Scatter(
    x=[ptm_position],
    y=['PTM'],
    mode='markers',
    marker=dict(size=12, color='olive'),
    name='PTM'
))

# Layout
fig.update_layout(
    height=500,
    title="Protein RCL Prediction with Feature Tracks",
    xaxis=dict(title='Residue Position', range=[0, sequence_length]),
    yaxis=dict(showticklabels=True),
    barmode='overlay',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.write_html("rcl_plot.html", auto_open=True)

fig.show()

