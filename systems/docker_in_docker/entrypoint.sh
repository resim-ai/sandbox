#!/bin/sh

set -e

mkdir -p /tmp/resim/outputs/gpu{0,1}

retry() {
    attempt=0
    until "$@"
    do
	attempt=$(( attempt + 1 ))
        echo "Waiting on $* attempt #$attempt"
        sleep 5

	if [ $attempt -gt 3 ]
	then
	    echo "Failed to initialize waiting on $*!"
	    exit 0
	fi
    done
}

echo "I see experience inputs:"
find "/tmp/resim/inputs/"


dockerd --iptables=false --ip6tables=false 2>/dev/null&

retry docker load -i /gpu-image.tar

# You could also use a compose here if you desire
nvidia-smi
nvidia-container-cli -k -d /dev/tty info

docker run -d --name gpu0 --runtime=nvidia --gpus='"device=0"' --privileged --volume "/tmp/resim/outputs/gpu0:/tmp/resim/outputs:rw" gpu-image:latest nvidia-smi
docker run -d --name gpu1 --runtime=nvidia --gpus='"device=1"' --privileged --volume "/tmp/resim/outputs/gpu1:/tmp/resim/outputs:rw" gpu-image:latest nvidia-smi

echo "Ran images!"

for CONTAINER in "gpu0" "gpu1"; do
    docker container wait "${CONTAINER}" > /dev/null
    docker logs "${CONTAINER}" > "/tmp/resim/outputs/${CONTAINER}-log.txt"
done
