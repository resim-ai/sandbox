#!/bin/bash

set -e

EXPDIR="$(dirname "$0")/experiences/local_experience/"
OUTDIR="$(pwd)"

# WARNING: This will not work if you run it from inside a docker container since
# volume paths are specified on the host system. We could get around this by
# creating temporary docker volumes and copying the inputs in and the outputs
# out, but we feel that this would overcomplicate this example.
docker run -it \
       --volume "${EXPDIR}:/tmp/resim/inputs" \
       --volume "${OUTDIR}:/tmp/resim/outputs" \
       hello_world:latest


# Run the metrics image. Note how the logs in OUTDIR are needed in the
# /tmp/resim/inputs/logs folder for this step.
#docker run -it \
#       --volume "${EXPDIR}:/tmp/resim/inputs/experience" \
#       --volume "${OUTDIR}:/tmp/resim/inputs/logs" \
#       --volume "${OUTDIR}:/tmp/resim/outputs" \
#       hello_world_metrics:latest
