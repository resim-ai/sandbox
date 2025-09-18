from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import OwlViTProcessor, OwlViTForObjectDetection


# literals
objects_to_detect = [["Car", "Truck", "Van"]]


class OwlModel:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )

    def plot_objects(self, image_filename):
        (boxes, scores, labels) = self.detect_objects(image_filename, 0.35)

        i = 0
        image = Image.open(image_filename)

        text = objects_to_detect[i]

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image)

        # Print detected objects and rescaled box coordinates
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
            )

            # Create a Rectangle patch
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )

            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add label and score
            ax.text(
                box[0],
                box[1] - 5,
                f"{text[label]}: {score.item():.2f}",
                color="red",
                fontsize=8,
            )

        plt.axis("off")
        plt.show()

    def detect_objects(self, image_filename, threshold: float):
        image = Image.open(image_filename)
        inputs = self.processor(
            text=objects_to_detect, images=image, return_tensors="pt"
        )
        outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])

        results = self.processor.post_process_object_detection(
            outputs, threshold, target_sizes=target_sizes
        )

        i = 0
        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )

        # Filter boxes by width and height
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = box.tolist()
            width = x1 - x0
            height = y1 - y0
            if width >= 20 and height >= 20:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label)

        return (
            torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4)),
            torch.tensor(filtered_scores),
            torch.tensor(filtered_labels),
        )
