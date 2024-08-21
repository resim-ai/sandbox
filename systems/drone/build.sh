
set -e

DRONE_DIR=$(dirname "$0")
cd "${DRONE_DIR}"

if [[ -n $VELOCITY_COST_OVERRIDE ]]; then
    VELOCITY_COST_OVERRIDE="--build-arg=VELOCITY_COST_OVERRIDE=$VELOCITY_COST_OVERRIDE"
fi
docker build -t drone_sim $VELOCITY_COST_OVERRIDE .
