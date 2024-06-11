#!/bin/bash
set -e


# Helper functions
_require() {
    if [[ -z "${!1}" ]]
    then
	echo "Environment variable $1 must be set!"
	exit 1
    fi
}

_set_default() {
    if [[ -z "${!1}" ]]
    then
	eval "${1}=${2}"
    fi
}

_print() {
    echo "${1}=${!1}"
}


LOCAL_TAG="hello_world:latest"
LOCAL_METRICS_TAG="hello_world_metrics:latest"

_require RESIM_SANDBOX_ECR
_require RESIM_SANDBOX_ECR_REPO
_require RESIM_SANDBOX_ECR_REGION
_require RESIM_SANDBOX_PROJECT
_require RESIM_SANDBOX_SYSTEM

_set_default RESIM_SANDBOX_BUILD_TAG_PREFIX "hello_world_"
_set_default RESIM_SANDBOX_METRICS_BUILD_TAG_PREFIX "hello_world_metrics_"
_set_default RESIM_SANDBOX_BUILD_BRANCH "hello_world_sandbox"
_set_default RESIM_SANDBOX_BUILD_VERSION "$(git rev-parse HEAD)"
_set_default RESIM_API_URL "https://api.resim.ai/v1"

echo "Running with..."
_print RESIM_SANDBOX_ECR       
_print RESIM_SANDBOX_ECR_REPO  
_print RESIM_SANDBOX_ECR_REGION
_print RESIM_SANDBOX_PROJECT
_print RESIM_SANDBOX_SYSTEM
_print RESIM_SANDBOX_BUILD_TAG_PREFIX
_print RESIM_SANDBOX_BUILD_BRANCH
_print RESIM_SANDBOX_BUILD_VERSION

echo "Ensuring the ReSim cli is installed..."
cd $(dirname "$0")
../../scripts/maybe_install_cli.sh

echo "Building the build image..."
./build.sh

echo "Building the metrics image..."
./metrics/build.sh

echo "Performing ECR Login..."
docker tag hello_world_metrics:latest 909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images:hello_world_metrics_lain
aws ecr get-login-password --region "us-east-1" \
    | docker login --username AWS --password-stdin "909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images"

echo "Pushing build image..."
_FULL_TAG="${RESIM_SANDBOX_ECR}/${RESIM_SANDBOX_ECR_REPO}:${RESIM_SANDBOX_BUILD_TAG_PREFIX}${RESIM_SANDBOX_BUILD_VERSION}"
docker tag "$AL_TAG}" "${_FULL_TAG}"
docker push "${_FULL_TAG}"

echo "Registering build with ReSim..."
resim builds create \
      --url "${RESIM_API_URL}" \
      --auth-url "${RESIM_AUTH_URL}" \
      --branch "${RESIM_SANDBOX_BUILD_BRANCH}" \
      --description "A ReSim sandbox build." \
      --image "${_FULL_TAG}" \
      --project "${RESIM_SANDBOX_PROJECT}" \
      --system "${RESIM_SANDBOX_SYSTEM}" \
      --version "${RESIM_SANDBOX_BUILD_VERSION}" \
      --auto-create-branch

echo "Pushing metrics image..."
_FULL_METRICS_TAG="${RESIM_SANDBOX_ECR}/${RESIM_SANDBOX_ECR_REPO}:${RESIM_SANDBOX_METRICS_BUILD_TAG_PREFIX}${RESIM_SANDBOX_BUILD_VERSION}"
docker tag "${LOCAL_METRICS_TAG}" "${_FULL_METRICS_TAG}"
docker push "${_FULL_METRICS_TAG}"

echo "Registering build with ReSim..."
resim metrics-builds create \
      --url "${RESIM_API_URL}" \
      --auth-url "${RESIM_AUTH_URL}" \
      --name "A ReSim sandbox metrics build." \
      --image "${_FULL_METRICS_TAG}" \
      --project "${RESIM_SANDBOX_PROJECT}" \
      --version "${RESIM_SANDBOX_BUILD_VERSION}"
