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
aws ecr get-login-password --region "${RESIM_SANDBOX_ECR_REGION}" \
    | docker login --username AWS --password-stdin "${RESIM_SANDBOX_ECR}"

echo "Pushing build image..."
_FULL_TAG="${RESIM_SANDBOX_ECR}/${RESIM_SANDBOX_ECR_REPO}:${RESIM_SANDBOX_BUILD_TAG_PREFIX}${RESIM_SANDBOX_BUILD_VERSION}"
docker tag "${LOCAL_TAG}" "${_FULL_TAG}"
docker push "${_FULL_TAG}"

echo "Registering build with ReSim..."
resim builds create \
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
      --name "A ReSim sandbox metrics build." \
      --image "${_FULL_METRICS_TAG}" \
      --project "${RESIM_SANDBOX_PROJECT}" \
      --version "${RESIM_SANDBOX_BUILD_VERSION}"
