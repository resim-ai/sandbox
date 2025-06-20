import plotly.graph_objects as go
from resim.metrics.proto import metrics_pb2 as mu
from resim.metrics.python import metrics_writer as rmw
from resim.metrics.python.metrics import (
  MetricStatus,
  MetricImportance
)

def add_summary_table_metric(
    writer: rmw.ResimMetricsWriter,
    total_detections: int,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    title: str = "Detection Summary",
) -> None:
    """
    Add a summary table to Resim metrics with detection stats (Total, TP, FP, FN).
    """
    headers = ["Statistic", "Value"]
    rows = [
        ["Total Detections", total_detections],
        ["True Positives", true_positives],
        ["False Positives", false_positives],
        ["False Negatives", false_negatives],
    ]

    # Transpose for Plotly
    stat_names = [r[0] for r in rows]
    stat_vals = [r[1] for r in rows]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=headers),
                cells=dict(values=[stat_names, stat_vals]),
            )
        ]
    )

    (
        writer.add_plotly_metric(name=title)
        .with_description("Summary detection statistics for OWLViT")
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_should_display(True)
        .with_blocking(False)
        .with_plotly_data(fig.to_json())
    )