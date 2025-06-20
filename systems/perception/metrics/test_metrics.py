from pathlib import Path
import pandas as pd
import ast
from typing import List, Dict, Tuple, Optional
from resim.metrics.python import metrics_writer as rmw



from metric_charts import add_summary_table_metric, add_precision_recall_curve
import numpy as np

IOU_THRESHOLD = 0.5

# --- Data class for bounding boxes ---
class BoundingBox:
    def __init__(self, bbox: List[float], cls: str, score: Optional[float] = None):
        self.xmin, self.ymin, self.xmax, self.ymax = bbox
        self.cls = cls
        self.score = score
        self.matched = False

    def iou(self, other: 'BoundingBox') -> float:
        xi1, yi1 = max(self.xmin, other.xmin), max(self.ymin, other.ymin)
        xi2, yi2 = min(self.xmax, other.xmax), min(self.ymax, other.ymax)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter_area == 0:
            return 0.0
        box1_area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        box2_area = (other.xmax - other.xmin) * (other.ymax - other.ymin)
        return inter_area / (box1_area + box2_area - inter_area)

# --- Parsing helpers ---
def parse_boxes(box_str: str, is_prediction: bool) -> List[BoundingBox]:
    box_list = ast.literal_eval(box_str)
    return [BoundingBox(b["bbox"], b["class"], b.get("score") if is_prediction else None) for b in box_list]

def load_csv(csv_path: str) -> Tuple[Dict[str, List[BoundingBox]], List[Tuple[str, BoundingBox]]]:
    df = pd.read_csv(csv_path)
    gt_dict, all_preds = {}, []

    for _, row in df.iterrows():
        fname = Path(row["filename"]).name
        gt_boxes = parse_boxes(row["gt_bbox"], is_prediction=False)
        pred_boxes = parse_boxes(row["model_bbox"], is_prediction=True)
        gt_dict[fname] = gt_boxes
        for box in pred_boxes:
            all_preds.append((fname, box))

    return gt_dict, all_preds

# --- Evaluation ---
def match_and_score(gt_dict: Dict[str, List[BoundingBox]], all_preds: List[Tuple[str, BoundingBox]]) -> Tuple[List[int], List[int], int]:
    all_preds.sort(key=lambda x: x[1].score or 0, reverse=True)
    tp, fp = [], []
    matched = {img: [False] * len(gt_list) for img, gt_list in gt_dict.items()}

    for img, pred_box in all_preds:
        gt_boxes = gt_dict.get(img, [])
        ious = [pred_box.iou(gt) for gt in gt_boxes]
        max_iou = max(ious, default=0.0)
        match_idx = ious.index(max_iou) if max_iou >= IOU_THRESHOLD else -1

        if match_idx != -1 and not matched[img][match_idx]:
            tp.append(1)
            fp.append(0)
            matched[img][match_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    total_gt = sum(len(v) for v in gt_dict.values())
    return tp, fp, total_gt

# --- PR Curve Calculation ---
def compute_pr_curve(tp: List[int], fp: List[int], total_gt: int) -> Tuple[List[float], List[float]]:
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall = tp_cum / (total_gt + 1e-8)
    return precision.tolist(), recall.tolist()

# --- Main driver ---
def run_test_metrics(writer: rmw.ResimMetricsWriter):
    csv_path = "/tmp/resim/inputs/logs/detections.csv"
    print("Loading the CSV")
    gt_dict, all_preds = load_csv(csv_path)
    
    print("Calculating IOU and matching detections")
    
    # tp, fp are both list of binaries containing either if they were true positive or false positive
    tp, fp, total_gt = match_and_score(gt_dict, all_preds)
    
    true_positives = sum(tp) # every 1 gets added
    false_positives = sum(fp)
    false_negatives = total_gt - true_positives # False Negatives = Ground truth boxes - true positives
    # Computing Precision and Recall
    precision, recall = compute_pr_curve(tp, fp, total_gt)
    
    add_summary_table_metric(writer, len(all_preds), true_positives, false_positives, false_negatives)
    
    print("Precision-Recall Curve:")
    for p, r in zip(precision, recall):
        print(f"Precision: {p:.3f}, Recall: {r:.3f}")
    if precision:
        print(f"\nFinal Precision@Recall=1: {precision[-1]:.3f}")
        
    add_precision_recall_curve(writer,precision,recall)
    print("Finished Running Test Metrics")