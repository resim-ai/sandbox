#!/bin/bash

# Edit these to point to folders on your host machine
HOST_INPUT_FOLDER="../experience/output"
HOST_OUTPUT_FOLDER="./output"

# Docker image name
IMAGE_NAME="perception_metrics_test"

# Run the container with volume mappings
docker run --platform linux/amd64 -it \
  -v "${HOST_INPUT_FOLDER}:/tmp/resim/inputs/logs" \
  -v "${HOST_OUTPUT_FOLDER}:/tmp/resim/outputs" \
  ${IMAGE_NAME}