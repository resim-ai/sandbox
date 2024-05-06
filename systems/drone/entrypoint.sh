#!/bin/bash

INPUTS="/tmp/resim/inputs"
OUTPUTS="/tmp/resim/outputs"


# Apply any parameters needed to the experience

c=`cat <<EOF
import os
import json
import pathlib
from google.protobuf import text_format
import resim.experiences.proto.experience_pb2 as Experience

def apply_params():
    params_path = pathlib.Path("${INPUTS}/parameters.json")
    if not params_path.is_file():
        return
    with open(params_path, "r") as f:
        params = json.load(f)
    if "velocity_cost" not in params:
        return

    with open("${INPUTS}/experience.sim", "r") as expfile:
        exp = text_format.Parse(expfile.read(), Experience.Experience())
    for movement_model in exp.dynamic_behavior.storyboard.movement_models:
        if movement_model.HasField("ilqr_drone"):
            movement_model.ilqr_drone.velocity_cost = eval(params["velocity_cost"])

    with open("${INPUTS}/experience.sim", "w") as expfile:
        expfile.write(str(exp))
apply_params()
EOF`
PYTHONPATH=/resim python3 -c "$c"

pushd /resim/resim_run.runfiles/resim_open_core/

resim/simulator/resim_run \
    --config "${INPUTS}/experience.sim" \
    --log "${OUTPUTS}/resim_log.mcap"

popd

pushd /resim/make_visualization_log.runfiles/resim_open_core/

resim/visualization/log/make_visualization_log \
    --log "${OUTPUTS}/resim_log.mcap" \
    --output "${OUTPUTS}/vis.mcap" \
    --world_glb "${INPUTS}/world.glb"

popd
