from pathlib import Path
import pandas as pd
import ast
from typing import List, Dict, Tuple, Optional
from resim.metrics.python import metrics_writer as rmw
from dataclasses import dataclass
from fp_event_creator import create_fp_event_v2


from metric_charts import *

IOU_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.05
OUT_PATH = Path("/tmp/resim/outputs")

@dataclass
class MatchResults:
    tp: List[int] # a vector of detection results with 1 if true positive and 0 if not
    fp: List[int] # a vector of detection results with 1 if false positive and 9 if not
    scores: List[float] # confidence scores for every result
    total_gt: int # Total detections present in the ground truth (needed for false negatives calc)
    
@dataclass
class SummaryMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

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
    '''
    LoadCSV returns a tuple containing 
    1. a dict {filename: [GT bounding box]}
    2. [(filename, pred_bbox)] - Each prediction is sepearately formed in a list so we can sort by confidence
    '''
    df = pd.read_csv(csv_path)
    gt_dict, all_preds = {}, []

    for _, row in df.iterrows():
        fname = Path(row["filename"])
        gt_boxes = parse_boxes(row["gt_bbox"], is_prediction=False)
        pred_boxes = parse_boxes(row["model_bbox"], is_prediction=True)
        gt_dict[fname] = gt_boxes
        for box in pred_boxes:
            all_preds.append((fname, box))

    return gt_dict, all_preds

# --- Evaluation ---
def match_and_score(
    gt_dict: Dict[str, List[BoundingBox]],
    all_preds: List[Tuple[str, BoundingBox]],
    writer: Optional[rmw.ResimMetricsWriter] = None,
    min_detection_conf: float = 0.45
) -> MatchResults:
    all_preds.sort(key=lambda x: x[1].score or 0, reverse=True)
    tp, fp, scores = [], [], []
    
    # mark every gt img as not matched
    matched = {img: [False] * len(gt_list) for img, gt_list in gt_dict.items()}

    for img, pred_box in all_preds:
        score = pred_box.score or 0.0
        scores.append(score)
        # Get the gt list for the image corresponding to the prediction
        gt_boxes = gt_dict.get(img, [])
        ious = [pred_box.iou(gt) for gt in gt_boxes]
        max_iou = max(ious, default=0.0)
        match_idx = ious.index(max_iou) if max_iou >= IOU_THRESHOLD else -1
        fp_num = 0
        # if we find a match and we did not previously find a match for that bbox, it is a TP
        if match_idx != -1 and not matched[img][match_idx]:
            tp.append(1)
            fp.append(0)
            matched[img][match_idx] = True
        else:
            tp.append(0)
            fp.append(1) # This happens for either duplicates or not matches. TODO - maybe handle duplicates differently
            
            fp_num += 1 
            if writer is not None and score >= min_detection_conf:
                # add_fp_image_event(writer, img)
                create_fp_event_v2(writer,img,OUT_PATH)



    total_gt = sum(len(v) for v in gt_dict.values())
    return MatchResults(tp=tp, fp=fp, scores=scores, total_gt=total_gt)

# --- PR Curve Calculation ---
def compute_pr_curve(match_result: MatchResults) -> Tuple[List[float], List[float]]:
    tp_cum, fp_cum = np.cumsum(match_result.tp), np.cumsum(match_result.fp)
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall = tp_cum / (match_result.total_gt + 1e-8)
    return precision.tolist(), recall.tolist()

def calculate_summary_stats(match_result: MatchResults, min_confidence: float = 0.45) -> SummaryMetrics:
    tp = sum(tp for tp, score in zip(match_result.tp, match_result.scores) if score >= min_confidence)
    fp = sum(fp for fp, score in zip(match_result.fp, match_result.scores) if score >= min_confidence)
    fn = match_result.total_gt - tp
    return SummaryMetrics(true_positives=tp, false_positives=fp, false_negatives=fn)

# --- Main driver ---
def run_test_metrics(writer: rmw.ResimMetricsWriter):
    csv_path = "/tmp/resim/inputs/logs/detections.csv"
    print("Loading the CSV")
    gt_dict, all_preds = load_csv(csv_path)
    
    print("Calculating IOU and matching detections")
    
    # tp, fp are both list of binaries containing either if they were true positive or false positive
    match_result = match_and_score(gt_dict, all_preds,writer,MIN_CONFIDENCE)
    
    print("Total Obstacles in the scene from Ground truth are:  ",match_result.total_gt)
    
    # Prompt to chatgpt:name this calculate_summary_stats(match_result) and name a class SummaryMetrics with the below three values
    summary_metrics = calculate_summary_stats(match_result, MIN_CONFIDENCE)
    
    
    # Computing Precision and Recall
    precision, recall = compute_pr_curve(match_result)
    
    add_summary_table_metric(writer, len(all_preds), summary_metrics.true_positives, summary_metrics.false_positives, summary_metrics.false_negatives)
    
    add_scalar_metrics(writer, summary_metrics.false_positives, summary_metrics.true_positives, summary_metrics.false_negatives)
    
    print("Precision-Recall Curve:")
    for p, r in zip(precision, recall):
        print(f"Precision: {p:.3f}, Recall: {r:.3f}")
    if precision:
        print(f"\nFinal Precision@Recall=1: {precision[-1]:.3f}")
        
    add_precision_recall_curve(writer,precision,recall)
    print("Finished Running Test Metrics")