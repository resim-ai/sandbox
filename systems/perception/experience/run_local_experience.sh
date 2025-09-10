#!/bin/bash

# Local test script that mounts samples folder as /dataset and ground_truth.csv as /tmp/resim/inputs/ground_truth.csv
HOST_INPUT_FOLDER="./samples"
HOST_OUTPUT_FOLDER="./output"

# Docker image name
IMAGE_NAME="perception_test"

# Run the container with volume mappings
# - Mount samples folder as /dataset (for images)
# - Mount ground_truth.csv as /tmp/resim/inputs/ground_truth.csv (for CSV file)
# - Mount output folder as /tmp/resim/outputs
docker run --platform linux/amd64 -it \
  -v "${HOST_INPUT_FOLDER}:/dataset" \
  -v "${HOST_INPUT_FOLDER}/ground_truth.csv:/tmp/resim/inputs/ground_truth.csv" \
  -v "${HOST_OUTPUT_FOLDER}:/tmp/resim/outputs" \
  ${IMAGE_NAME}
