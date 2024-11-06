#!/bin/sh

set -e

cd $(dirname "$0")


IMAGE="hello-world"
docker pull "${IMAGE}:latest"
docker save "${IMAGE}:latest" -o "${IMAGE}.tar"

docker build . -t docker-in-docker-test-image

rm "${IMAGE}.tar"
