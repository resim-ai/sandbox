#!/bin/sh

set -e

cd $(dirname "$0")

./build.sh

UUID=$(uuidgen)
INPUTS_DIR="/tmp/resim/${UUID}/inputs"
OUTPUTS_DIR="/tmp/resim/${UUID}/outputs"

mkdir -p "${INPUTS_DIR}"
touch "${INPUTS_DIR}/some_input.txt"
mkdir "${INPUTS_DIR}/some_input_folder/"
touch "${INPUTS_DIR}/some_input_folder/some_other_input.exe"
touch "${INPUTS_DIR}/some_input_folder/hooray.jif"

docker run --privileged \
       --volume "${INPUTS_DIR}:/tmp/resim/inputs:ro" \
       --volume "${OUTPUTS_DIR}:/tmp/resim/outputs:rw" \
       docker-in-docker-test-image

echo "I see outputs:"
find "${OUTPUTS_DIR}"
