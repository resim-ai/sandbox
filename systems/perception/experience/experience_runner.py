import json
import pandas as pd
import os
from pathlib import Path
from owl_model import OwlModel
import argparse
import cv2
import time
import numpy as np
from resim.metrics.python.emissions import Emitter

# Path to a single image for testing - so model can be cached in the docker image
test_image_path = "/tmp/sample_image.png"

model = OwlModel()

emitter = Emitter(config_path=Path("/workspaces/sandbox/.resim/metrics/config.yml"))

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        float: IoU score between 0 and 1
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

def calculate_detection_metrics(model_boxes, ground_truth_boxes, iou_threshold=0.5):
    """Calculate precision, recall, F1, and IoU metrics.
    
    Args:
        model_boxes: List of model detection boxes
        ground_truth_boxes: List of ground truth boxes
        iou_threshold: IoU threshold for considering a match
    
    Returns:
        dict: Metrics including precision, recall, F1, IoU scores, etc.
    """
    num_detections = len(model_boxes)
    num_gt_objects = len(ground_truth_boxes)
    
    if num_detections == 0 and num_gt_objects == 0:
        return {
            'num_true_positives': 0,
            'num_false_positives': 0,
            'num_false_negatives': 0,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
            'avg_iou': 1.0,
            'max_iou': 1.0,
            'min_iou': 1.0,
            'iou_scores': []
        }
    
    if num_detections == 0:
        return {
            'num_true_positives': 0,
            'num_false_positives': 0,
            'num_false_negatives': num_gt_objects,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'avg_iou': 0.0,
            'max_iou': 0.0,
            'min_iou': 0.0,
            'iou_scores': []
        }
    
    if num_gt_objects == 0:
        return {
            'num_true_positives': 0,
            'num_false_positives': num_detections,
            'num_false_negatives': 0,
            'precision': 0.0,
            'recall': 1.0,
            'f1_score': 0.0,
            'avg_iou': 0.0,
            'max_iou': 0.0,
            'min_iou': 0.0,
            'iou_scores': []
        }
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((num_detections, num_gt_objects))
    for i, model_box in enumerate(model_boxes):
        for j, gt_box in enumerate(ground_truth_boxes):
            if 'bbox' in model_box and 'bbox' in gt_box:
                iou_matrix[i, j] = calculate_iou(model_box['bbox'], gt_box['bbox'])
    
    # Find best matches using greedy assignment
    matched_detections = set()
    matched_gt = set()
    iou_scores = []
    
    # Sort by IoU score (descending)
    matches = []
    for i in range(num_detections):
        for j in range(num_gt_objects):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))
    
    matches.sort(reverse=True)
    
    for iou_score, det_idx, gt_idx in matches:
        if det_idx not in matched_detections and gt_idx not in matched_gt:
            matched_detections.add(det_idx)
            matched_gt.add(gt_idx)
            iou_scores.append(iou_score)
    
    num_true_positives = len(matched_detections)
    num_false_positives = num_detections - num_true_positives
    num_false_negatives = num_gt_objects - num_true_positives
    
    # Calculate metrics
    precision = num_true_positives / num_detections if num_detections > 0 else 0.0
    recall = num_true_positives / num_gt_objects if num_gt_objects > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate IoU statistics
    if iou_scores:
        avg_iou = np.mean(iou_scores)
        max_iou = np.max(iou_scores)
        min_iou = np.min(iou_scores)
    else:
        avg_iou = 0.0
        max_iou = 0.0
        min_iou = 0.0
    
    return {
        'num_true_positives': num_true_positives,
        'num_false_positives': num_false_positives,
        'num_false_negatives': num_false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_iou': avg_iou,
        'max_iou': max_iou,
        'min_iou': min_iou,
        'iou_scores': iou_scores
    }

def create_visualization_image(image_path, model_boxes, ground_truth_boxes, output_path):
    """Create a visualization image with bounding boxes from model and ground truth."""
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Colors: Green for ground truth, Red for model predictions
    gt_color = (0, 255, 0)  # Green for ground truth
    model_color = (0, 0, 255)  # Red for model predictions
    
    # Draw ground truth bounding boxes
    for gt_box in ground_truth_boxes:
        if isinstance(gt_box, dict) and 'bbox' in gt_box:
            bbox = gt_box['bbox']
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), gt_color, 2)
                cv2.putText(image, 'GT', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1)
    
    # Draw model prediction bounding boxes
    for model_box in model_boxes:
        if isinstance(model_box, dict) and 'bbox' in model_box:
            bbox = model_box['bbox']
            score = model_box.get('score', 0.0)
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), model_color, 2)
                cv2.putText(image, f'Model: {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, model_color, 1)
    
    # Save the visualization image
    cv2.imwrite(output_path, image)
    print(f"Visualization saved: {output_path}")

def create_enhanced_visualization(image_path, model_boxes, ground_truth_boxes, 
                                detection_ids, gt_ids, proposal_to_gt, gt_to_proposal, output_path):
    """Create an enhanced visualization with IDs and match indicators."""
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Colors
    gt_color = (0, 255, 0)  # Green for ground truth
    model_color = (0, 0, 255)  # Red for model predictions
    match_color = (255, 255, 0)  # Yellow for matches
    no_match_color = (128, 128, 128)  # Gray for no matches
    
    # Draw ground truth bounding boxes with IDs
    for i, gt_box in enumerate(ground_truth_boxes):
        if isinstance(gt_box, dict) and 'bbox' in gt_box:
            bbox = gt_box['bbox']
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                # Choose color based on match status
                if i in gt_to_proposal:
                    color = match_color
                    status = "MATCHED"
                else:
                    color = no_match_color
                    status = "NO MATCH"
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f'GT {gt_ids[i]} ({status})', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw model prediction bounding boxes with IDs
    for i, model_box in enumerate(model_boxes):
        if isinstance(model_box, dict) and 'bbox' in model_box:
            bbox = model_box['bbox']
            score = model_box.get('score', 0.0)
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                # Choose color based on match status
                if i in proposal_to_gt:
                    color = match_color
                    status = "MATCHED"
                else:
                    color = no_match_color
                    status = "NO MATCH"
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f'Det {detection_ids[i]} ({status})', (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(image, f'Conf: {score:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Save the enhanced visualization image
    cv2.imwrite(output_path, image)
    print(f"Enhanced visualization saved: {output_path}")

def get_experience_config() -> dict:
    """Get the experience configuration containing image and ground truth locations."""
    test_config_path = Path("/tmp/resim/test_config.json")
    if test_config_path.exists():
        with open(test_config_path, "r") as f:
            test_config = json.load(f)
            # Parse experienceLocations to determine image and ground truth
            locations = test_config.get("experienceLocations", [])
            image_location = None
            ground_truth_location = None
            
            for location in locations:
                if location.endswith('.png'):
                    image_location = location
                elif location.endswith('.csv'):
                    ground_truth_location = location
            
            return {
                "imageLocation": image_location,
                "groundTruthLocation": ground_truth_location
            }
    else:
        # Default fallback for testing
        return {
            "imageLocation": "/experiences/sample_image.png",
            "groundTruthLocation": "/experiences/sample_image-truth.csv"
        }


# The code that runs the model on a single image
def analyze_single_image(image_path, ground_truth_path, output_path):
    """Analyze a single image with its ground truth data."""
    
    start_time = time.time()
    
    # Check if image file exists
    if not os.path.isfile(image_path):
        print(f"Image file does not exist: {image_path}")
        exit(1)
    
    # Check if ground truth file exists
    if not os.path.isfile(ground_truth_path):
        print(f"Ground truth file does not exist: {ground_truth_path}")
        exit(1)
    
    # Read ground truth data
    df = pd.read_csv(ground_truth_path)
    
    # Load image for metrics
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    image_height, image_width = image.shape[:2]
    filename = os.path.basename(image_path)
    
    threshold_score = 0.01  # a small delta value
    
    # Run the model on the image
    boxes, scores, labels = model.detect_objects(image_path, threshold_score)
    print(f"Running model on image: {image_path}: detections {len(boxes)}")
    
    # Populate the results
    model_boxes = [
        {
            "class": "Vehicle",
            "bbox": [round(i, 2) for i in box.tolist()],
            "score": round(score.item(), 3),
        }
        for box, score, label in zip(boxes, scores, labels, strict=True)
    ]
    
    # Parse ground truth data for visualization
    ground_truth_boxes = []
    if not df.empty:
        for _, row in df.iterrows():
            if 'bbox' in row and not pd.isna(row['bbox']):
                try:
                    # Parse the bbox data (assuming it's JSON string)
                    bbox_value = row['bbox']
                    if isinstance(bbox_value, str):
                        bbox_data = json.loads(bbox_value)
                    else:
                        bbox_data = bbox_value
                    if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                        ground_truth_boxes.append({'bbox': bbox_data})
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, try to use the raw data
                    bbox_value = row['bbox']
                    if isinstance(bbox_value, (list, tuple)) and len(bbox_value) >= 4:
                        ground_truth_boxes.append({'bbox': bbox_value})
    
    # Create visualization image
    visualization_path = f"{output_path}_visualization.png"
    create_visualization_image(image_path, model_boxes, ground_truth_boxes, visualization_path)
    
    # Calculate metrics
    num_detections = len(model_boxes)
    num_gt_objects = len(ground_truth_boxes)
    detection_ratio = num_detections / num_gt_objects if num_gt_objects > 0 else 0.0
    
    # Calculate confidence statistics
    if scores:
        avg_confidence = float(np.mean(scores))
        max_confidence = float(np.max(scores))
        min_confidence = float(np.min(scores))
        detection_scores = [float(score) for score in scores]
    else:
        avg_confidence = 0.0
        max_confidence = 0.0
        min_confidence = 0.0
        detection_scores = []
    
    # Calculate IoU-based detection metrics
    detection_metrics = calculate_detection_metrics(model_boxes, ground_truth_boxes, iou_threshold=0.5)
    
    # Extract environmental conditions from ground truth CSV
    weather = df['weather'].iloc[0] if 'weather' in df.columns and not df.empty else "unknown"
    road_type = df['road_type'].iloc[0] if 'road_type' in df.columns and not df.empty else "unknown"
    lighting = df['lighting'].iloc[0] if 'lighting' in df.columns and not df.empty else "unknown"
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Generate IDs and match detections to ground truth
    try:
        # Generate unique IDs for detections and ground truth
        detection_ids = [f"det_{i:03d}" for i in range(num_detections)]
        gt_ids = [f"gt_{i:03d}" for i in range(num_gt_objects)]
        
        # Calculate IoU matrix for matching
        iou_matrix = np.zeros((num_detections, num_gt_objects))
        for i, model_box in enumerate(model_boxes):
            for j, gt_box in enumerate(ground_truth_boxes):
                if 'bbox' in model_box and 'bbox' in gt_box:
                    iou_matrix[i, j] = calculate_iou(model_box['bbox'], gt_box['bbox'])
        
        # Find best matches using greedy assignment
        matched_proposals = set()
        matched_gt = set()
        proposal_to_gt = {}  # Maps proposal_id to gt_id
        gt_to_proposal = {}  # Maps gt_id to proposal_id
        
        # Sort by IoU score (descending)
        matches = []
        for i in range(num_detections):
            for j in range(num_gt_objects):
                if iou_matrix[i, j] >= 0.5:  # IoU threshold
                    matches.append((iou_matrix[i, j], i, j))
        
        matches.sort(reverse=True)
        
        for iou_score, prop_idx, gt_idx in matches:
            if prop_idx not in matched_proposals and gt_idx not in matched_gt:
                matched_proposals.add(prop_idx)
                matched_gt.add(gt_idx)
                proposal_to_gt[prop_idx] = gt_idx
                gt_to_proposal[gt_idx] = prop_idx
        
        # Create enhanced visualization with IDs
        create_enhanced_visualization(image_path, model_boxes, ground_truth_boxes, 
                                    detection_ids, gt_ids, proposal_to_gt, gt_to_proposal,
                                    f"{output_path}_enhanced_visualization.png")
        
        # Emit ground truth objects
        for i, gt_box in enumerate(ground_truth_boxes):
            emitter.emit("gt_object", {
                "filename": filename,
                "type": "Car",
                "x1": float(gt_box["bbox"][0]),
                "y1": float(gt_box["bbox"][1]),
                "x2": float(gt_box["bbox"][2]),
                "y2": float(gt_box["bbox"][3])
            })
        
        # Emit detections
        for i, model_box in enumerate(model_boxes):
            emitter.emit("detection", {
                "filename": filename,
                "x1": float(model_box["bbox"][0]),
                "y1": float(model_box["bbox"][1]),
                "x2": float(model_box["bbox"][2]),
                "y2": float(model_box["bbox"][3]),
                "score": float(model_box.get("score", 0.0))
            })
        
        # Emit image metadata
        emitter.emit("image", {
            "filename": filename,
            "number_of_objects": num_gt_objects,
            "road_type": road_type,
            "weather": weather,
            "lighting": lighting
        })
        
        # Emit visualization data
        emitter.emit("visualization", {
            "filename": filename,
            "image_path": f"{output_path}_enhanced_visualization.png",
            "visualization_type": "enhanced"
        })
        
        # Emit object events for each detection/GT pair
        for i, model_box in enumerate(model_boxes):
            if i in proposal_to_gt:
                # True positive event
                gt_idx = proposal_to_gt[i]
                iou_score = iou_matrix[i, gt_idx]
                status = "PASSED" if iou_score >= 0.7 else "FAIL_WARN"
                emitter.emit("object", {
                    "name": f"Detection {detection_ids[i]} matched GT {gt_ids[gt_idx]}",
                    "description": f"IoU: {iou_score:.3f}, Confidence: {model_box.get('score', 0.0):.3f}",
                    "status": status,
                    "tags": ["true_positive", f"iou_{iou_score:.2f}", f"conf_{model_box.get('score', 0.0):.2f}"]
                })
            else:
                # False positive event
                emitter.emit("object", {
                    "name": f"Detection {detection_ids[i]} - No Match",
                    "description": f"Confidence: {model_box.get('score', 0.0):.3f}",
                    "status": "FAIL_BLOCK",
                    "tags": ["false_positive", f"conf_{model_box.get('score', 0.0):.2f}"]
                })
        
        for i, gt_box in enumerate(ground_truth_boxes):
            if i not in gt_to_proposal:
                # False negative event
                emitter.emit("object", {
                    "name": f"GT {gt_ids[i]} - No Detection",
                    "description": "Ground truth object not detected",
                    "status": "FAIL_BLOCK",
                    "tags": ["false_negative"]
                })
        
        print(f"Emitted metrics for {filename}: {num_detections} detections, {num_gt_objects} GT objects, {len(matched_proposals)} matches")
        
    except Exception as e:
        print(f"Error emitting metrics: {e}")
    
    # Create result with ground truth and model predictions
    result = {
        "filename": os.path.basename(image_path),
        "gt_bbox": df.to_json(orient='records') if not df.empty else "[]",
        "model_bbox": json.dumps(model_boxes),
    }
    
    # Save results
    output_csv = f"{output_path}.csv"
    pd.DataFrame([result]).to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run model on a test image")
    args = parser.parse_args()
    
    if args.test:
        print(f"Running model on test image: {test_image_path}")
        boxes, scores, labels = model.detect_objects(test_image_path, threshold=0.0)
        for box, score, label in zip(boxes, scores, labels, strict=True):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected: bbox={box}, score={score.item():.2f}, label={label}")
        exit()

    # Get experience configuration with image and ground truth locations
    config = get_experience_config()
    image_path = config.get("imageLocation")
    ground_truth_path = config.get("groundTruthLocation")
    
    # Check if we have both required locations
    if not image_path:
        print("Error: No image location found in config")
        exit(1)
    if not ground_truth_path:
        print("Error: No ground truth location found in config")
        exit(1)
    
    # Output folder
    output_folder = "/tmp/resim/outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Name the file to write to
    output_path_base = os.path.join(output_folder, "detections")

    print(f"Processing image: {image_path}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Output CSV file: {output_path_base}")

    analyze_single_image(image_path, ground_truth_path, output_path_base)
