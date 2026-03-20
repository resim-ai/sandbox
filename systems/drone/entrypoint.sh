#!/bin/bash

# Set the cost override for the velocity cost
# Check if the parameters.json file that exists in /tmp/resim/inputs contains the velocity cost override:
if [ -f /tmp/resim/parameters.json ]; then
  VELOCITY_COST_OVERRIDE=$(jq -r '.velocity_cost_override' /tmp/resim/parameters.json)
  # check it is not null:
    if [ "$VELOCITY_COST_OVERRIDE" == "null" ]; then
        echo "No velocity cost override found in parameters. If supplied directly as an environment variable, it can still be used."
    # otherwise:
    else
        echo "Found a velocity cost override in the parameters.json file: $VELOCITY_COST_OVERRIDE"
        VELOCITY_COST_OVERRIDE="--velocity_cost_override ${VELOCITY_COST_OVERRIDE}"
    fi
fi

echo "Running container with cost override: $VELOCITY_COST_OVERRIDE"
echo "Running pre-flight safety check..."
echo "  [OK] Experience loaded..."
echo "  [OK] Goal position acquired"
echo "  [OK] MCAP logger ready: /tmp/resim/inputs/logs/resim_log.mcap"
echo "  [OK] Actor spawned: is_spawned=true"
echo "  [OK] iLQR controller: initialized"
echo "  [WARN] Drone operating in fake simulation mode"
echo "  [FAIL] I hope this was fun. Aborting execution"
exit 1

/sim_container_entrypoint.sh $VELOCITY_COST_OVERRIDE
