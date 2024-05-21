#!/bin/bash

set -e

METRICS_DIR=$(dirname "$0")
cd "${METRICS_DIR}"

rm -rf python_libs || true
mkdir python_libs

pushd ../../../open_core

bazel build @resim_open_core//resim/metrics/proto:validate_metrics_proto
bazel build @resim_open_core//resim/metrics/python:metrics_writer
bazel build @resim_open_core//resim/actor/state/proto:observable_state_proto_py
bazel build @resim_open_core//resim/transforms/python:se3_python.so

OUT_DIR="$(bazel info bazel-bin)/external/resim_open_core~/"
DEP_DIR="$(bazel info output_base)/external/resim_open_core~/"

popd


# Copy all python libs over
c=`cat <<EOF
import shutil
import os

for d in ("${OUT_DIR}", "${DEP_DIR}"):
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith(".py") or f.endswith(".so"):
                target_dir = os.path.join(
                    "python_libs",
                    os.path.relpath(root, d))
                os.makedirs(target_dir, exist_ok=True)
                shutil.copyfile(os.path.join(root, f),
                                os.path.join(target_dir, f))
EOF`
python3 -c "$c"

docker build -t drone_sim_metrics .
