import json
import pandas as pd
import os
from pathlib import Path
from owl_model import OwlModel
import argparse
import cv2
import numpy as np

# Path to a single image for testing - so model can be cached in the docker image
test_image_path = "/tmp/sample_image.png"

model = OwlModel()

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
        for box, score, label in zip(boxes, scores, labels)
    ]
    
    # Parse ground truth data for visualization
    ground_truth_boxes = []
    if not df.empty:
        for _, row in df.iterrows():
            if 'bbox' in row and pd.notna(row['bbox']):
                try:
                    # Parse the bbox data (assuming it's JSON string)
                    bbox_data = json.loads(row['bbox']) if isinstance(row['bbox'], str) else row['bbox']
                    if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                        ground_truth_boxes.append({'bbox': bbox_data})
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, try to use the raw data
                    if isinstance(row['bbox'], (list, tuple) and len(row['bbox']) >= 4):
                        ground_truth_boxes.append({'bbox': row['bbox']})
    
    # Create visualization image
    visualization_path = f"{output_path}_visualization.png"
    create_visualization_image(image_path, model_boxes, ground_truth_boxes, visualization_path)
    
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
        for box, score, label in zip(boxes, scores, labels):
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
