import plotly.graph_objects as go
from resim.metrics.python import metrics_writer as rmw
import resim.metrics.python.metrics_utils as muu
from typing import List
from resim.metrics.python.metrics import (
    MetricStatus,
    MetricImportance,
    ExternalFileMetricsData,
)
from resim.metrics.resim_style import (
    resim_plotly_style,
    RESIM_TURQUOISE,
)

import uuid


def add_precision_recall_curve(
    writer: rmw.ResimMetricsWriter, precision: list, recall: list
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines+markers",
            name="PR Curve",
            line=dict(color=RESIM_TURQUOISE, width=2),
        )
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Recall"),
        yaxis=dict(title="Precision"),
        showlegend=True,
        legend=dict(orientation="h", y=1.0, x=0.5, xanchor="center", yanchor="bottom"),
    )

    resim_plotly_style(fig)
    (
        writer.add_plotly_metric("Precision-Recall Curve")
        .with_description("Precision-Recall curve for camera detections.")
        .with_blocking(False)
        .with_plotly_data(str(fig.to_json()))
        .with_importance(MetricImportance.HIGH_IMPORTANCE)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
    )


def add_scalar_metrics(
    writer: rmw.ResimMetricsWriter, false_positives, true_positives, false_negatives
):
    (
        writer.add_scalar_metric("False Positives")
        .with_description("False Positives in the sequence from the model")
        .with_importance(MetricImportance.HIGH_IMPORTANCE)
        .with_value(false_positives)
        .with_unit("")
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
    )

    (
        writer.add_scalar_metric("True Positives")
        .with_description("True Positives in the sequence from the model")
        .with_importance(MetricImportance.HIGH_IMPORTANCE)
        .with_value(true_positives)
        .with_unit("")
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
    )

    (
        writer.add_scalar_metric("False Negatives")
        .with_description("False Negatives in the sequence from the model")
        .with_importance(MetricImportance.HIGH_IMPORTANCE)
        .with_value(false_negatives)
        .with_unit("")
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
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

    resim_plotly_style(fig)
    fig.update_layout(template="plotly")

    (
        writer.add_plotly_metric(name=title)
        .with_description("Summary detection statistics for OWLViT")
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_should_display(True)
        .with_blocking(False)
        .with_plotly_data(fig.to_json())
    )


def add_fp_image_event(writer: rmw.ResimMetricsWriter, filename: str):
    """
    This function will add the image to the required location when a false positive is
    detected.
    """
    EVENT_TIMESTAMP = muu.Timestamp(secs=0, nanos=0)
    # Start by creating the image metric - will need the bounding box as well
    # hack - to get timestamp
    writer.add_event(name=f"False Positive  {uuid.uuid4().hex}").with_description(
        "Unmatched with ground truth or duplicate"
    ).with_tags(["false"]).with_status(
        MetricStatus.PASSED_METRIC_STATUS
    ).with_importance(MetricImportance.HIGH_IMPORTANCE).with_relative_timestamp(
        EVENT_TIMESTAMP
    ).with_metrics(
        [
            # metrics must have unique names
            writer.add_text_metric(name="False Positive File")
            .with_text(f"{filename}")
            .with_description("File name")
            .with_status(MetricStatus.PASSED_METRIC_STATUS)
            .with_should_display(False)
            .with_importance(MetricImportance.MEDIUM_IMPORTANCE)
            .is_event_metric()
        ]
    )


def add_fp_image_list(
    writer: rmw.ResimMetricsWriter, img_list: List[ExternalFileMetricsData]
):
    (
        writer.add_image_list_metric(name="All False Positives")
        .with_description("All false positives with score > min threshold")
        .with_status(MetricStatus.FAIL_WARN_METRIC_STATUS)
        .with_importance(MetricImportance.LOW_IMPORTANCE)
        .with_should_display(True)
        .with_blocking(False)
        .with_image_list_data(img_list)
    )
