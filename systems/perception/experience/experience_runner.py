import json
import pandas as pd
import os
from pathlib import Path
from owl_model import OwlModel
import argparse

# Path to a single image for testing - so model can be cached in the docker image
test_image_path = "/tmp/sample_image.png"

model = OwlModel()

# Global variable for input path addendum
input_path_addendum = None


def get_experience_location() -> str:
    test_config_path = Path("/tmp/resim/test_config.json")
    if test_config_path.exists():
        with open(test_config_path, "r") as f:
            test_config = json.load(f)
            return test_config["experienceLocation"]
    else:
        return str(Path("/tmp/resim/inputs/ground_truth.csv"))


def detect_input_path_addendum(filename: str) -> str:
    """Detect the correct addendum for input paths by checking file existence."""
    global input_path_addendum

    if input_path_addendum is not None:
        return input_path_addendum

    # Check if file exists in /tmp/resim/inputs
    inputs_path = os.path.join("/tmp/resim/inputs", filename)
    if os.path.isfile(inputs_path):
        input_path_addendum = "/tmp/resim/inputs"
        return input_path_addendum

    # Check if file exists in /dataset
    dataset_path = os.path.join("/dataset", filename)
    if os.path.isfile(dataset_path):
        input_path_addendum = "/dataset"
        return input_path_addendum

    # Default to /tmp/resim/inputs if neither exists
    input_path_addendum = "/tmp/resim/inputs"
    return input_path_addendum


# The code that runs the model on the dataset
def analyze_sequence_v2(input_csv, output_path):
    df = pd.read_csv(input_csv)

    evaluated_rows = []

    for idx, row in df.iterrows():
        img_path = row["filename"]
        if not img_path.startswith("/tmp/resim/inputs") and not img_path.startswith(
            "/dataset"
        ):
            base_path = detect_input_path_addendum(img_path)
            img_path = os.path.join(base_path, img_path)

        # exit early if the file doesnt exist
        if not os.path.isfile(img_path):
            print("File does not  exist!")
            exit(1)

        threshold_score = 0.01  # a small delta value

        # Run the model
        boxes, scores, labels = model.detect_objects(img_path, threshold_score)
        print(f" Running model on image: {img_path}: detections {len(boxes)}")

        # populate the results
        model_boxes = [
            {
                "class": "Vehicle",
                "bbox": [round(i, 2) for i in box.tolist()],
                "score": round(score.item(), 3),
            }
            for box, score, label in zip(boxes, scores, labels)
        ]

        evaluated_rows.append(
            {
                "filename": row["filename"],
                "gt_bbox": row["gt_bbox"],
                "model_bbox": json.dumps(model_boxes),
            }
        )

    output_csv = f"{output_path}.csv"
    pd.DataFrame(evaluated_rows).to_csv(output_csv, index=False)
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

    input_csv = get_experience_location()
    # Output folder
    output_folder = "/tmp/resim/outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Name the file to write to
    output_path_base = os.path.join(output_folder, "detections")

    print(f"Output CSV file is {output_path_base}")

    analyze_sequence_v2(input_csv, output_path_base)
