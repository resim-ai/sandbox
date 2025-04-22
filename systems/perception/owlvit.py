import json
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "./systems/perception/a2d2/camera_lidar_semantic_bboxes/20180807_145028/camera/cam_front_center/20180807145028_camera_frontcenter_000028313.png"
image = Image.open(url)
texts = [["a van", "a car", "a street sign"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(outputs=outputs, threshold=0.35, target_sizes=target_sizes)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Create figure and axes
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(image)

# Print detected objects and rescaled box coordinates
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

    # Create a Rectangle patch
    rect = patches.Rectangle(
        (box[0], box[1]), 
        box[2] - box[0], 
        box[3] - box[1],
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    # Add label and score
    ax.text(
        box[0], box[1]-5,
        f'{text[label]}: {score.item():.2f}',
        color='red',
        fontsize=8
    )

# Load and parse the JSON file
with open("./systems/perception/a2d2/camera_lidar_semantic_bboxes/20180807_145028/label3D/cam_front_center/20180807145028_label3D_frontcenter_000028313.json") as f:
    label_data = json.loads(f.read())

# Draw each 2D bounding box from the JSON
for box_key, box_data in label_data.items():
    bbox = box_data["2d_bbox"]
    if box_data["id"] == 2:
        continue
    
    # Create rectangle patch for the 2D bbox
    # bbox format is [x1, y1, x2, y2]
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor='blue',
        facecolor='none'
    )
    
    # Add the patch to the axes
    ax.add_patch(rect)
    
    # Add label
    ax.text(
        bbox[0], bbox[1]-5,
        f'{box_data["class"]}: {box_data["id"]}',
        color='blue',
        fontsize=8
    )

plt.axis('off')
plt.show()
