#!/bin/bash

if [[ -n $VELOCITY_COST_OVERRIDE ]]; then
    VELOCITY_COST_OVERRIDE="--velocity_cost_override ${VELOCITY_COST_OVERRIDE}"
fi


/sim_container_entrypoint.sh $VELOCITY_COST_OVERRIDE
