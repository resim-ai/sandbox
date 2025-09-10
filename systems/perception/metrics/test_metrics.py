from pathlib import Path
import pandas as pd
import ast
import os
from typing import List, Dict, Tuple, Optional
from resim.metrics.python import metrics_writer as rmw
from dataclasses import dataclass
from bounding_box import BoundingBox
from fp_event_creator import create_fp_event_v2
from fn_event_creator import create_fn_event
from resim.metrics.python.metrics import ExternalFileMetricsData


from metric_charts import *

IOU_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.15
OUT_PATH = Path("/tmp/resim/outputs")
IN_ROOT_PATH = Path("/tmp/resim/inputs")

# Global variable for input path addendum
input_path_addendum = None

def detect_input_path_addendum(filename: str) -> str:
    """Detect the correct addendum for input paths by checking file existence."""
    global input_path_addendum
    
    if input_path_addendum is not None:
        return input_path_addendum
    
    # Check if file exists in /tmp/resim/inputs/experience
    experience_path = os.path.join("/tmp/resim/inputs/experience", filename)
    if os.path.isfile(experience_path):
        input_path_addendum = "/tmp/resim/inputs/experience"
        return input_path_addendum
    
    # Check if file exists in /dataset
    dataset_path = os.path.join("/dataset", filename)
    if os.path.isfile(dataset_path):
        input_path_addendum = "/dataset"
        return input_path_addendum
    
    # Default to /tmp/resim/inputs/experience if neither exists
    input_path_addendum = "/tmp/resim/inputs/experience"
    return input_path_addendum

@dataclass
class MatchResults:
    tp: List[int] # a vector of detection results with 1 if true positive and 0 if not
    fp: List[int] # a vector of detection results with 1 if false positive and 9 if not
    scores: List[float] # confidence scores for every result
    total_gt: int # Total detections present in the ground truth (needed for false negatives calc)
    unmatched_gt: Dict[Path, List[BoundingBox]] 
    fp_images: List[ExternalFileMetricsData]
    
@dataclass
class SummaryMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int


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
        relative_path = Path(row["filename"])
        

        # If path is absolute and under /tmp/resim/inputs, make it relative to that root
        if relative_path.is_absolute() and str(relative_path).startswith(str(IN_ROOT_PATH)):
            relative_path = relative_path.relative_to(IN_ROOT_PATH)

        # Use path detection to determine the correct base path
        base_path = detect_input_path_addendum(str(relative_path))
        fname = Path(base_path) / relative_path
        gt_boxes = parse_boxes(row["gt_bbox"], is_prediction=False)
        pred_boxes = parse_boxes(row["model_bbox"], is_prediction=True)
        gt_dict[fname] = gt_boxes
        for box in pred_boxes:
            all_preds.append((fname, box))

    return gt_dict, all_preds


def compile_unmatched_gt_boxes(
    gt_dict: Dict[Path, List[BoundingBox]],
    matched: Dict[Path, List[bool]]
) -> Dict[Path, List[BoundingBox]]:
    """
    Extracts all unmatched ground truth bounding boxes.

    Args:
        gt_dict: Mapping of image path → list of ground truth boxes.
        matched: Mapping of image path → list of booleans indicating matched status for each GT box.

    Returns:
        Dictionary mapping image path → list of unmatched bounding boxes.
    """
    unmatched_gt = {}
    for img, gt_list in gt_dict.items():
        unmatched_boxes = [
            box for idx, box in enumerate(gt_list)
            if not matched[img][idx]
        ]
        if unmatched_boxes:
            unmatched_gt[img] = unmatched_boxes
    return unmatched_gt

# --- Evaluation ---
def match_and_score(
    gt_dict: Dict[str, List[BoundingBox]],
    all_preds: List[Tuple[str, BoundingBox]],
    writer: Optional[rmw.ResimMetricsWriter] = None,
    min_detection_conf: float = 0.45
) -> MatchResults:
    all_preds.sort(key=lambda x: x[1].score or 0, reverse=True)
    tp, fp, scores = [], [], []
    fp_images = [] # for false positive image carousel
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
                print(f"False positive detected at img: {img}")
                img_data = create_fp_event_v2(writer,img,OUT_PATH,pred_box)
                fp_images.append(img_data)
                
            else:
                print(f"Score on img : {img} is : {score}")


    unmatched_gt = compile_unmatched_gt_boxes(gt_dict,matched)
    total_gt = sum(len(v) for v in gt_dict.values())
    return MatchResults(tp=tp, fp=fp, scores=scores, total_gt=total_gt, unmatched_gt=unmatched_gt, fp_images=fp_images)

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
    print("This is the latest run")
    print("Loading the CSV")
    gt_dict, all_preds = load_csv(csv_path)
    
    print("Calculating IOU and matching detections")
    
    # tp, fp are both list of binaries containing either if they were true positive or false positive
    match_result = match_and_score(gt_dict, all_preds,writer,MIN_CONFIDENCE)
    
    print("Total Obstacles in the scene from Ground truth are:  ",match_result.total_gt)
    
    # Log events for all false negatives
    # For each unmatched GT box, create a false negative event
    for img_path, unmatched_boxes in match_result.unmatched_gt.items():
        # Gather all predictions for this image (used to show context in event)
        pred_boxes = [box for (fname, box) in all_preds if fname == img_path]

        
        print(f"False negative(s) detected at img: {img_path}")
        create_fn_event(writer, str(img_path), OUT_PATH, unmatched_boxes, pred_boxes)
        
    # All false positives
    if match_result.fp_images:
        add_fp_image_list(writer,match_result.fp_images)
        
    
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
    