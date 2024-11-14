#!/bin/sh

set -e

cd $(dirname "$0")


IMAGE="nvidia/cuda:12.6.2-base-ubuntu24.04"
ALIAS="gpu-image"
docker pull "${IMAGE}"
docker tag  "${IMAGE}" "${ALIAS}:latest"
docker save "${ALIAS}:latest" -o "${ALIAS}.tar"

docker build . -t docker-in-docker-test-image

rm "${ALIAS}.tar"
