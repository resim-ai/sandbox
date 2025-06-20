import json
from PIL import Image
import pandas as pd
import os
from owl_model import OwlModel
import argparse

# Path to a single image for testing - so model can be cached in the docker image
test_image_path = '/tmp/sample_image.png'

# Thresholds to sweep over
thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30,0.25,0.20,0.15,0.10,0.05]
model = OwlModel()
def analyze_sequence(input_csv, output_path_base, threshold):
    df = pd.read_csv(input_csv)
    evaluated_rows = []

    for idx, row in df.iterrows():
        img_path = row['filename']
        if not os.path.isfile(img_path):
            print("File does not  exist!")
            exit(1)

        boxes, scores, labels = model.detect_objects(img_path, threshold)

        model_boxes = []
        print(f"[{threshold:.2f}] Running model on image: {img_path}: detections {len(boxes)}")
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            model_boxes.append({
                'class': 'Vehicle',
                'bbox': box,
                'score': round(score.item(), 3)
            })

        evaluated_rows.append({
            'filename': row['filename'],
            'gt_bbox': row['gt_bbox'],
            'model_bbox': json.dumps(model_boxes)
        })

    # Convert threshold to integer filename suffix (e.g., 0.05 -> '05', 0.95 -> '95')
    threshold_suffix = f"{int(threshold * 100):02d}"
    output_csv = f"{output_path_base}_thresh_{threshold_suffix}.csv"
    pd.DataFrame(evaluated_rows).to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")

def analyze_sequence_v2(input_csv, output_path):
    df = pd.read_csv(input_csv)

    evaluated_rows = []

    for idx,row in df.iterrows():
        img_path = row['filename']

        # exit early if the file doesnt exist
        if not os.path.isfile(img_path):
            print("File does not  exist!")
            exit(1)
        
        threshold_score = 0.01 # a small delta value

        # Run the model
        boxes, scores, labels = model.detect_objects(img_path, threshold_score)
        print(f" Running model on image: {img_path}: detections {len(boxes)}")

        # populate the results
        model_boxes = [
                {
                    'class': 'Vehicle',
                    'bbox': [round(i, 2) for i in box.tolist()],
                    'score': round(score.item(), 3)
                }
                for box, score, label in zip(boxes, scores, labels)
            ]

        evaluated_rows.append({
            'filename': row['filename'],
            'gt_bbox': row['gt_bbox'],
            'model_bbox': json.dumps(model_boxes)
        })

    output_csv = f"{output_path}.csv"
    pd.DataFrame(evaluated_rows).to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")
        
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run model on a test image')
    parser.add_argument('--old', action='store_true', help='Run model with separate thresholds outputs')
    args = parser.parse_args()
    if args.test:

        print(f"Running model on test image: {test_image_path}")
        boxes, scores, labels = model.detect_objects(test_image_path, threshold=0.0)
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected: bbox={box}, score={score.item():.2f}, label={label}")
        exit()

    input_csv = '/tmp/resim/inputs/ground_truth.csv'
    # Output folder
    output_folder = '/tmp/resim/outputs'
    os.makedirs(output_folder, exist_ok=True)
    
    # Name the file to write to
    output_path_base = os.path.join(output_folder, "detections")
    
    print(f"Output CSV file is {output_path_base}")

    if args.old:
        for thresh in thresholds:
            analyze_sequence(input_csv, output_path_base,thresh)
    else: 
        analyze_sequence_v2(input_csv, output_path_base)
