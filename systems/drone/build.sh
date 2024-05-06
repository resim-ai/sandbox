
set -e

DRONE_DIR=$(dirname "$0")
cd "${DRONE_DIR}"

rm -rf bin || true
mkdir bin

pushd ../../open_core

bazel build @resim_open_core//resim/simulator:resim_run
RESIM_RUN_RUNFILES_DIR="$(bazel info bazel-bin)/external/resim_open_core/resim/simulator/resim_run.runfiles/"


bazel build @resim_open_core//resim/visualization/log:make_visualization_log
MAKE_VISUALIZATION_LOG_RUNFILES_DIR="$(bazel info bazel-bin)/external/resim_open_core/resim/visualization/log/make_visualization_log.runfiles/"

bazel build @resim_open_core//resim/experiences/proto:experience_proto_py
EXP_PROTO_DIR="$(bazel info bazel-bin)/external/resim_open_core/"

popd

cp -Lr "${RESIM_RUN_RUNFILES_DIR}" bin/
cp -Lr "${MAKE_VISUALIZATION_LOG_RUNFILES_DIR}" bin/

# Copy all python libs over
c=`cat <<EOF
import shutil
import os

d = "${EXP_PROTO_DIR}"
for root, dirs, files in os.walk(d):
    for f in files:
        if f.endswith(".py") or f.endswith(".so"):
            target_dir = os.path.join(
                "bin",
                os.path.relpath(root, d))
            os.makedirs(target_dir, exist_ok=True)
            shutil.copyfile(os.path.join(root, f),
                            os.path.join(target_dir, f))
EOF`
python3 -c "$c"

docker build -t drone_sim .
