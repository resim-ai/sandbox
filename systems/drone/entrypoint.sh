#!/bin/bash

pushd /resim/resim_run.runfiles/resim_open_core/

resim/simulator/resim_run \
    --config /tmp/resim/inputs/experience.sim \
    --log /tmp/resim/outputs/resim_log.mcap

popd
