import cv2
from pathlib import Path
from bounding_box import BoundingBox

def draw_bbox(image, bbox: BoundingBox, color=(0, 0, 255), thickness=2):
    start = (int(bbox.xmin), int(bbox.ymin))
    end = (int(bbox.xmax), int(bbox.ymax))
    cv2.rectangle(image, start, end, color, thickness)

def draw_score(image, bbox: BoundingBox, font_scale=0.5, color=(255, 255, 255), bg_color=(0, 0, 255)):
    if bbox.score is None:
        return

    text = f"Score: {bbox.score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(bbox.xmin), max(int(bbox.ymin) - 5, th + 5)

    cv2.rectangle(image, (x, y - th - 2), (x + tw, y + baseline), bg_color, cv2.FILLED)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def annotate_image_with_prediction(image_path: Path, bbox: BoundingBox):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image at {image_path}")
    draw_bbox(image, bbox)
    draw_score(image, bbox)
    return image

def write_image_with_bbox(image_path: Path, dest_path: Path, bbox: BoundingBox):
    image = annotate_image_with_prediction(image_path, bbox)
    cv2.imwrite(str(dest_path), image)