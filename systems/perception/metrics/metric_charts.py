import plotly.graph_objects as go
from resim.metrics.proto import metrics_pb2 as mu
from resim.metrics.python import metrics_writer as rmw
from resim.metrics.python.metrics import (
  SeriesMetricsData,
  MetricStatus,
  MetricImportance
)
import numpy as np

def add_precision_recall_curve(writer: rmw.ResimMetricsWriter,precision: list,recall: list):
    precision_data = SeriesMetricsData(
        name="camera_precision_data",
        series=np.array(precision),
        unit=""
    )
    recall_data = SeriesMetricsData(
        name="camera_recall_data",
        series=np.array(recall),
        unit=""
    )
    status_data = SeriesMetricsData(
        name="camera_pr_statuses",
        series=np.array([MetricStatus.PASSED_METRIC_STATUS] * len(precision)),
        unit=""
    )
    (
        writer.add_line_plot_metric(name="Precision-Recall Curve")
        .with_description("Precision-Recall curve for camera detections")
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_should_display(True)
        .with_blocking(False)
        .append_series_data(recall_data, precision_data, "precision")
        .append_statuses_data(status_data)
        .with_x_axis_name("Recall")
        .with_y_axis_name("Precision")
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