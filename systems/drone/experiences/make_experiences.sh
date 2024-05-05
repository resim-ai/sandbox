#!/bin/bash

set -e

DIR=$(dirname "$0")

cd "${DIR}"

OPEN_CORE_DIR="${DIR}/../../../open_core/"
pushd "${OPEN_CORE_DIR}"

bazel build @resim_open_core//resim/experiences/proto:experience_proto_py
OUT_DIR="$(bazel info bazel-bin)/external/resim_open_core/"

popd

PYTHONPATH="${OUT_DIR}" python3 make_experiences.py
