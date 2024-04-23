#!/bin/bash
set -e

_require() {
    if [[ -z "${!1}" ]]
    then
	echo "Environment variable $1 must be set!"
	exit 1
    fi
}

which uuidgen > /dev/null || (echo "uuidgen is required!" && exit 1)

_require RESIM_SANDBOX_S3_PREFIX
_require RESIM_SANDBOX_PROJECT

echo "Ensuring the ReSim cli is installed..."
cd $(dirname "$0")
../../scripts/maybe_install_cli.sh


echo "Generating and pushing demo experiences..."
NUM_EXPERIENCES=10
EXPERIENCES_DIRECTORY="/tmp/$(uuidgen)/"
mkdir "${EXPERIENCES_DIRECTORY}"

for i in $(seq $NUM_EXPERIENCES); do
    # Create the experience locally
    ID=$(uuidgen)
    EXPERIENCE_DIRECTORY="${EXPERIENCES_DIRECTORY}/${ID}"
    mkdir "${EXPERIENCE_DIRECTORY}"
    touch "${EXPERIENCE_DIRECTORY}/experience_${ID}.sim"
    touch "${EXPERIENCE_DIRECTORY}/my_other_config.yaml"

    # Copy it up to S3
    EXPERIENCE_PREFIX="${RESIM_SANDBOX_S3_PREFIX}/${ID}"
    aws s3 cp --recursive "${EXPERIENCE_DIRECTORY}" "${EXPERIENCE_PREFIX}"

    # Register it with ReSim
    resim experiences create \
	  --description "Demo experience for hello world" \
	  --location "${EXPERIENCE_PREFIX}" \
	  --name "Hello World Experience ${ID}" \
	  --project "${RESIM_SANDBOX_PROJECT}"
done
