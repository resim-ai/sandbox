set -e

EXPDIR="$(dirname "$0")/experiences/local_experience/"
OUTDIR="$(pwd)"

docker run -it \
       --volume "${EXPDIR}:/tmp/resim/inputs" \
       --volume "${OUTDIR}:/tmp/resim/outputs" \
       drone_sim:latest
