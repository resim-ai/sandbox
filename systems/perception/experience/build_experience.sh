#!/bin/bash

# Set image name and tag
IMAGE_NAME="perception_test"
TAG="latest"

# The large dataset files are on S3. Contact resim team to get access.

# Create dataset directory if it doesn't exist (to prevent Docker build failure)
if [ ! -d "./dataset" ]; then
    echo "Creating empty dataset directory..."
    mkdir -p ./dataset
fi

# Build the Docker image for linux/amd64 using Dockerfile.exp
docker buildx build \
  --platform linux/amd64 \
  -t ${IMAGE_NAME}:${TAG} \
  .