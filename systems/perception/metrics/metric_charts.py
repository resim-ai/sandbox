import plotly.graph_objects as go
from resim.metrics.proto import metrics_pb2 as mu
from resim.metrics.python import metrics_writer as rmw
import resim.metrics.python.metrics_utils as muu
from resim.metrics.python.metrics import (
  SeriesMetricsData,
  MetricStatus,
  MetricImportance
)
import numpy as np
import uuid
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
    
def add_scalar_metrics(writer, false_positives, true_positives,false_negatives):
    writer.add_scalar_metric("False Positives")\
        .with_description("False Positives in the sequence from the model")\
        .with_importance(MetricImportance.HIGH_IMPORTANCE)\
        .with_value(false_positives)\
        .with_unit("")\
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        
    writer.add_scalar_metric("True Positives")\
        .with_description("True Positives in the sequence from the model")\
        .with_importance(MetricImportance.HIGH_IMPORTANCE)\
        .with_value(true_positives)\
        .with_unit("")\
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        
    writer.add_scalar_metric("False Negatives")\
        .with_description("False Negatives in the sequence from the model")\
        .with_importance(MetricImportance.HIGH_IMPORTANCE)\
        .with_value(false_negatives)\
        .with_unit("")\
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        
    

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
    
def add_fp_image_event(writer: rmw.ResimMetricsWriter,filename: str):
    '''
        This function will add the image to the required location when a false positive is
        detected. 
    '''
    EVENT_TIMESTAMP = muu.Timestamp(secs=0, nanos=0)
    # Start by creating the image metric - will need the bounding box as well
    # hack - to get timestamp
    writer.add_event(
            name=f"False Positive  {uuid.uuid4().hex}"
        ).with_description("Unmatched with ground truth or duplicate")\
        .with_tags(['false'])\
        .with_status(MetricStatus.PASSED_METRIC_STATUS)\
        .with_importance(MetricImportance.HIGH_IMPORTANCE)\
        .with_relative_timestamp(EVENT_TIMESTAMP)\
        .with_metrics(
            [
                # metrics must have unique names
                writer.add_text_metric(
                    name=f"False Positive File"
                )
                .with_text(
                    f"{filename}"
                )
                .with_description("File name")
                .with_status(MetricStatus.PASSED_METRIC_STATUS)
                .with_should_display(False)
                .with_importance(MetricImportance.MEDIUM_IMPORTANCE)
                .is_event_metric()
            ]
        )
    
    
    pass