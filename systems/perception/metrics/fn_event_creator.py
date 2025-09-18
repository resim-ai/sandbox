from pathlib import Path
from typing import List
from bounding_box import BoundingBox
from resim.metrics.python import metrics_writer as rmw
from resim.metrics.python import metrics_utils as mu
from resim.metrics.python.metrics import (
    ExternalFileMetricsData,
    ImageMetric,
    MetricStatus,
    MetricImportance,
)
import cv2

# Constants
FN_GT_IMAGE_DIRNAME = "fn_images/gt"
FN_PRED_IMAGE_DIRNAME = "fn_images/pred"


def save_gt_image_with_bboxes(
    image_path: str, out_dir: Path, gt_boxes: List[BoundingBox]
) -> ExternalFileMetricsData:
    """
    Save the ground truth image with all unmatched GT boxes in red.
    """
    image_path = Path(image_path)
    dest_dir = out_dir / FN_GT_IMAGE_DIRNAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image at {image_path}")

    # Draw each GT box in red
    for box in gt_boxes:
        start = (int(box.xmin), int(box.ymin))
        end = (int(box.xmax), int(box.ymax))
        cv2.rectangle(image, start, end, color=(0, 0, 255), thickness=2)

    # Use original filename (no hash)
    dest_path = dest_dir / image_path.name
    cv2.imwrite(str(dest_path), image)

    return ExternalFileMetricsData(
        name=f"FN GT Image: {dest_path.name}",
        filename=str(dest_path.relative_to(out_dir)),
    )


def save_pred_image_with_bboxes(
    image_path: str, out_dir: Path, pred_boxes: List[BoundingBox]
) -> ExternalFileMetricsData:
    """
    Save the prediction image with all predicted boxes (green).
    """
    image_path = Path(image_path)
    dest_dir = out_dir / FN_PRED_IMAGE_DIRNAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image at {image_path}")

    # Draw each predicted box in green
    for box in pred_boxes:
        start = (int(box.xmin), int(box.ymin))
        end = (int(box.xmax), int(box.ymax))
        cv2.rectangle(image, start, end, color=(0, 255, 0), thickness=2)

    # Save annotated image using original filename
    dest_path = dest_dir / image_path.name
    cv2.imwrite(str(dest_path), image)

    return ExternalFileMetricsData(
        name=f"FN Pred Image: {dest_path.name}",
        filename=str(dest_path.relative_to(out_dir)),
    )


def create_fn_image_metrics(
    writer: rmw.ResimMetricsWriter,
    img_path: Path,
    gt_image_data: ExternalFileMetricsData,
    pred_image_data: ExternalFileMetricsData,
) -> List[ImageMetric]:
    """
    Create two image metrics for FN: one for the unmatched GT and one for the predictions.
    Returns a list of metrics.
    """
    gt_metric = (
        writer.add_image_metric(name="False Negative Ground Truth")
        .with_description(f"Unmatched Ground truth in image {img_path.stem}")
        .with_status(MetricStatus.FAIL_WARN_METRIC_STATUS)
        .with_importance(MetricImportance.MEDIUM_IMPORTANCE)
        .with_should_display(True)
        .with_blocking(False)
        .with_image_data(gt_image_data)
        .is_event_metric()
    )

    pred_metric = (
        writer.add_image_metric(name="False Negative Predictions")
        .with_description(f"All predictions for image {img_path.stem} (missed GTs)")
        .with_status(MetricStatus.FAIL_WARN_METRIC_STATUS)
        .with_importance(MetricImportance.LOW_IMPORTANCE)
        .with_should_display(True)
        .with_blocking(False)
        .with_image_data(pred_image_data)
        .is_event_metric()
    )

    return [gt_metric, pred_metric]


def create_fn_event(
    writer: rmw.ResimMetricsWriter,
    image_path: str,
    out_dir: Path,
    gt_boxes: List[BoundingBox],
    pred_boxes: List[BoundingBox],
):
    """
    Main entry point to create a false negative event.
    Saves annotated GT/pred images, builds metrics, and attaches event to Resim.
    """
    img_path = Path(image_path)

    # Save annotated images
    gt_image_data = save_gt_image_with_bboxes(image_path, out_dir, gt_boxes)
    pred_image_data = save_pred_image_with_bboxes(image_path, out_dir, pred_boxes)

    # Create image metrics
    image_metrics = create_fn_image_metrics(
        writer, img_path, gt_image_data, pred_image_data
    )

    # Attach event
    (
        writer.add_event(name=f"False Negative {img_path.stem}")
        .with_description(f"Missed ground truth in image {img_path.stem}")
        .with_status(MetricStatus.FAIL_WARN_METRIC_STATUS)
        .with_importance(MetricImportance.MEDIUM_IMPORTANCE)
        .with_relative_timestamp(mu.Timestamp(secs=0, nanos=0))
        .with_metrics(image_metrics)
        .with_tags(["false_negative"])
    )
