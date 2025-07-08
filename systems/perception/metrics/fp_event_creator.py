import shutil
from pathlib import Path
import uuid
import resim.metrics.python.metrics_utils as mu
from resim.metrics.python import metrics_writer as rmw
from bounding_box import BoundingBox
from image_writer import write_image_with_bbox
from resim.metrics.python.metrics import (
  ExternalFileMetricsData,      
  MetricStatus,
  MetricImportance
)

METRIC_IMAGES_DIRNAME = "metric_images"

def save_image_with_bbox(image_path: str, out_dir: Path, pred_box: BoundingBox) -> ExternalFileMetricsData:
    '''
    Function takes the path to the input image, and then embeds the bounding box and saves it to file and generates
    an returns an `ExternalFileMetricsData` object, which can be used to add the image to the ReSim Web UI for events
    '''
    dest_dir = out_dir / METRIC_IMAGES_DIRNAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    new_name = f"{uuid.uuid4().hex}_{Path(image_path).name}"
    dest_path = dest_dir / new_name

    # call OpenCV logic from image_writer
    write_image_with_bbox(Path(image_path), dest_path, pred_box)

    return ExternalFileMetricsData(
        name=f"FP Image: {dest_path.name}",
        filename=str(dest_path.relative_to(out_dir)),
    )


def create_fp_event_v2(writer: rmw.ResimMetricsWriter, image_path: str, out_dir: Path, pred_box: BoundingBox):
    """
    Create a Resim event for a false positive detection by attaching the raw image.

    Args:
        writer: The ResimMetricsWriter instance.
        image_path: Absolute path to the false positive image.
        out_dir: Base output directory for Resim metrics (e.g., /tmp/resim/outputs)
    """
    # image_data = save_image_for_metric(image_path, out_dir)
    image_data = save_image_with_bbox(image_path, out_dir, pred_box)
    img_path = Path(image_path)
    hash = uuid.uuid4().hex
    metric = create_image_metric(writer, img_path, image_data ,hash)
    create_event(writer, img_path, metric, hash)


def save_image_for_metric(image_path: str, out_dir: Path) -> ExternalFileMetricsData:
    """
    Copy the image to the metrics output directory (flat), return ExternalFileMetricsData.

    Args:
        image_path: Path to the original image.
        out_dir: Base output directory for Resim metrics.

    Returns:
        ExternalFileMetricsData pointing to the copied image.
    """
    image_path = Path(image_path)
    dest_dir = out_dir / METRIC_IMAGES_DIRNAME
    dest_dir.mkdir(exist_ok=True, parents=True)

    new_name = f"{uuid.uuid4().hex}_{image_path.name}"
    dest_path = dest_dir / new_name
    shutil.copyfile(image_path, dest_path)

    return ExternalFileMetricsData(
        name=f"FP Image: {dest_path.name}",
        filename=str(dest_path.relative_to(out_dir)),
    )


def create_image_metric(
    writer: rmw.ResimMetricsWriter,
    img_path: Path,
    image_data: ExternalFileMetricsData,
    event_hash: str,
):
    """
    Create an image list metric for the given image.
    """
    return (
        writer.add_image_metric(name=f"False Positive img {img_path.stem} - {event_hash}")
        .with_description(f"False positive detected at image {img_path.stem}")
        .with_status(MetricStatus.FAIL_WARN_METRIC_STATUS)
        .with_importance(MetricImportance.MEDIUM_IMPORTANCE)
        .with_should_display(True)
        .with_blocking(False)
        .with_image_data(image_data)
        .is_event_metric()
    )


def create_event(
    writer: rmw.ResimMetricsWriter,
    img_path: Path,
    image_metric,
    event_hash: str,
):
    """
    Finalize and attach the event to the writer.
    """
    (
        writer.add_event(name=f"False Positive {img_path.stem} - {event_hash}")
        .with_description(f"Unmatched detection in image {img_path.stem}")
        .with_status(MetricStatus.FAIL_WARN_METRIC_STATUS)
        .with_importance(MetricImportance.MEDIUM_IMPORTANCE)
        .with_relative_timestamp(mu.Timestamp(secs=0, nanos=0))
        .with_metrics([image_metric])
        .with_tags(["false_positive"])
    )