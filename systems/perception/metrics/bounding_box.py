from typing import List, Optional


class BoundingBox:
    def __init__(self, bbox: List[float], cls: str, score: Optional[float] = None):
        self.xmin, self.ymin, self.xmax, self.ymax = bbox
        self.cls = cls
        self.score = score
        self.matched = False

    def iou(self, other: "BoundingBox") -> float:
        xi1, yi1 = max(self.xmin, other.xmin), max(self.ymin, other.ymin)
        xi2, yi2 = min(self.xmax, other.xmax), min(self.ymax, other.ymax)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter_area == 0:
            return 0.0
        box1_area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        box2_area = (other.xmax - other.xmin) * (other.ymax - other.ymin)
        return inter_area / (box1_area + box2_area - inter_area)
