
set -e

DRONE_DIR=$(dirname "$0")
cd "${DRONE_DIR}"

rm -rf bin || true
mkdir bin

pushd ../../open_core

bazel build @resim_open_core//resim/simulator:resim_run
RESIM_RUN_RUNFILES_DIR="$(bazel info bazel-bin)/external/resim_open_core/resim/simulator/resim_run.runfiles/"


popd

cp -Lr "${RESIM_RUN_RUNFILES_DIR}" bin/

ls bin/

docker build -t drone_sim .
