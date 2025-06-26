from flask import Blueprint, render_template
import plotly.graph_objects as go
from evaluate import evaluate_sequence
import numpy as np

main = Blueprint("main", __name__)

@main.route("/")
def highlight():
    # text = "abcdefghij"
    # indices_to_highlight = {1, 4, 7}

    #sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    sequence = "MPSSVSWGILLLAGLCCLVPVSLAEDPQGDAAQKTDTSHHDQDHPTFNKITPNLAEFAFSLYRQLAHQSNSTNIFFSPVSIATAFAMLSLGTKADTHDEILEGLNFNLTEIPEAQIHEGFQELLRTLNQPDSQLQLTTGNGLFLSEGLKLVDKFLEDVKKLYHSEAFTVNFGDTEEAKKQINDYVEKGTQGKIVDLVKELDRDTVFALVNYIFFKGKWERPFEVKDTEEEDFHVDQVTTVKVPMMKRLGMFNIQHCKKLSSWVLLMKYLGNATAIFFLPDEGKLQHLENELTHDIITKFLENEDRRSASLHLPKLSITGTYDLKSVLGQLGITKVFSNGADLSGVTEEAPLKLSKAVHKAVLTIDEKGTEAAGAMFLEAIPMSIPPEVKFNKPFVFLMIEQNTKSPLFMGKVVNPTQK"
    # probs = [0.01, 0.02, 0.05, 0.12, 0.3, 0.4, 0.55, 0.7, 0.82, 0.9,
    #         0.8, 0.7, 0.65, 0.55, 0.52, 0.49, 0.4, 0.3, 0.2, 0.1,
    #         0.01, 0.01, 0.03, 0.05, 0.12, 0.2, 0.35, 0.51, 0.65, 0.7,
    #         0.8, 0.9, 0.95]
    # pred_indices = [6,7,8,9,10,11,12,13,14,27,28,29,30,31,32]
    probs = evaluate_sequence(sequence)
    pred_indices = np.where(probs > 0.5)[0]

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(sequence))),
        y=probs,
        marker_color='red',
        mode='lines+markers',
        hovertext=[f"{aa} (Pos {i+1})" for i, aa in enumerate(sequence)],
        hoverinfo='text+y'
    ))
    fig.add_hline(y=0.5, line=dict(color="black", dash="dash"))

    # Add aligned sequence text as x-axis tick labels
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(sequence))),
            ticktext=list(sequence),
            tickfont=dict(family='Courier New, monospace', size=18)
        ),
        yaxis=dict(range=[0, 1.05]),
        title="Interactive RCL Prediction Plot",
        height=400
    )
    plot_html = fig.to_html(full_html=False)

    highlighted_text = ""
    for i, char in enumerate(sequence):
        if i in pred_indices:
            highlighted_text += f'<span class="highlight">{char}</span>'
        else:
            highlighted_text += char

    return render_template("results.html", highlighted_text=highlighted_text, plot_html=plot_html)