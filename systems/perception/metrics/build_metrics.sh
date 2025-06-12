#!/bin/bash

# Set image name and tag
IMAGE_NAME="perception_metrics_test"
TAG="latest"

# Build the Docker image for linux/amd64 using Dockerfile.exp
docker buildx build \
  --platform linux/amd64 \
  -f dockerfile.met \
  -t ${IMAGE_NAME}:${TAG} \
  .