from flask import Blueprint, render_template, request
import plotly.graph_objects as go
from evaluate import evaluate_sequence
import numpy as np

main = Blueprint("main", __name__)

@main.route("/", methods=["GET", "POST"])
def input():
    return render_template("home.html")

@main.route("/results", methods=["POST"])
def results():
    # text = "abcdefghij"
    # indices_to_highlight = {1, 4, 7}

    # sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    probs = [0.01, 0.02, 0.05, 0.12, 0.3, 0.4, 0.55, 0.7, 0.82, 0.9,
            0.8, 0.7, 0.65, 0.55, 0.52, 0.49, 0.4, 0.3, 0.2, 0.1,
            0.01, 0.01, 0.03, 0.05, 0.12, 0.2, 0.35, 0.51, 0.65, 0.7,
            0.8, 0.9, 0.95]
    pred_indices = [6,7,8,9,10,11,12,13,14,27,28,29,30,31,32]

    sequence = "MPSSVSWGILLLAGLCCLVPVSLAEDPQGDAAQKTDTSHHDQDHPTFNKITPNLAEFAFSLYRQLAHQSNSTNIFFSPVSIATAFAMLSLGTKADTHDEILEGLNFNLTEIPEAQIHEGFQELLRTLNQPDSQLQLTTGNGLFLSEGLKLVDKFLEDVKKLYHSEAFTVNFGDTEEAKKQINDYVEKGTQGKIVDLVKELDRDTVFALVNYIFFKGKWERPFEVKDTEEEDFHVDQVTTVKVPMMKRLGMFNIQHCKKLSSWVLLMKYLGNATAIFFLPDEGKLQHLENELTHDIITKFLENEDRRSASLHLPKLSITGTYDLKSVLGQLGITKVFSNGADLSGVTEEAPLKLSKAVHKAVLTIDEKGTEAAGAMFLEAIPMSIPPEVKFNKPFVFLMIEQNTKSPLFMGKVVNPTQK"
    # sequence = request.form.get("sequence")
    # probs = evaluate_sequence(sequence)
    # pred_indices = np.where(probs > 0.5)[0]

    # Plot
    fig = go.Figure()

    # Add invisible scatter for hover text
    fig.add_trace(go.Scatter(
        x=pred_indices,
        y=[0.5] * len(pred_indices),
        mode="markers",
        marker=dict(size=0.1, color="rgba(0,0,0,0)"),  # invisible
        hoverinfo="text",
        text=[f"{sequence[i]} (Pos {i+1})" for i in pred_indices]
    ))

    # Add individual red rectangles
    shapes = []
    for idx in pred_indices:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=idx - 0.5,
                x1=idx + 0.5,
                y0=0,
                y1=1,
                fillcolor="rgba(255, 0, 0, 1.0)",
                line=dict(width=0)
            )
        )

    # Add baseline
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=-0.5,
            x1=len(sequence) - 0.5,
            y0=0.5,
            y1=0.5,
            line=dict(color="lightgray", width=2)
        )
    )

    # Update layout
    fig.update_layout(
        shapes=shapes,
        annotations=[
            dict(
                x= - .5,
                y=0.5,
                xref="x",
                yref="paper",
                text="PRED RCL",
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right",
            )
        ],
        xaxis=dict(
            range=[-0.5, len(sequence) - 0.5],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            visible=False
        ),
        height=40,
        margin=dict(t=10, b=10, l=80, r=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=None
    )
    plot_html = fig.to_html(full_html=False, config={
        "responsive": True,
        "displayModeBar": False,
        "scrollZoom": True,
        "doubleClick": "reset"
    })

    highlighted_text = ""
    for i, char in enumerate(sequence):
        if i in pred_indices:
            highlighted_text += f'<span class="highlight">{char}</span>'
        else:
            highlighted_text += char

    return render_template("results.html", highlighted_text=highlighted_text, plot_html=plot_html, sequence=sequence, rcl_indices=pred_indices)