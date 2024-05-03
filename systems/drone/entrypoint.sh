#!/bin/bash

INPUTS="/tmp/resim/inputs"
OUTPUTS="/tmp/resim/outputs"

pushd /resim/resim_run.runfiles/resim_open_core/

resim/simulator/resim_run \
    --config "${INPUTS}/experience.sim" \
    --log "${OUTPUTS}/resim_log.mcap"

popd

pushd /resim/make_visualization_log.runfiles/resim_open_core/

resim/visualization/log/make_visualization_log \
    --log "${OUTPUTS}/resim_log.mcap" \
    --output "${OUTPUTS}/vis.mcap"

popd
