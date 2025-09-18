#!/bin/bash

# Edit these to point to folders on your host machine
BATCH_CONFIG_FILE="./batch_metrics_config.json"
CONTAINER_BATCH_LOCATION="/tmp/resim/inputs/batch_metrics_config.json"
HOST_OUTPUT_FOLDER="./output"

# Docker image name
IMAGE_NAME="perception_metrics_test"

# Run the container with volume mappings
docker run --platform linux/amd64 -it \
  -v "${BATCH_CONFIG_FILE}:${CONTAINER_BATCH_LOCATION}"\
  -v "${HOST_OUTPUT_FOLDER}:/tmp/resim/outputs" \
  ${IMAGE_NAME}