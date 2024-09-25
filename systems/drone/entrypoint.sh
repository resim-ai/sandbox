#!/bin/bash

# Set the cost override for the velocity cost
# Check if the parameters.json file that exists in /tmp/resim/inputs contains the velocity cost override:
if [ -f /tmp/resim/inputs/parameters.json ]; then
  VELOCITY_COST_OVERRIDE=$(jq -r '.velocity_cost_override' /tmp/resim/inputs/parameters.json)
  echo "Found a velocity cost override in the parameters.json file: $VELOCITY_COST_OVERRIDE"
fi

echo "Running container with cost override: $VELOCITY_COST_OVERRIDE"
/sim_container_entrypoint.sh $VELOCITY_COST_OVERRIDE
