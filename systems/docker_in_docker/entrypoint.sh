#!/bin/sh

set -e

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

retry docker load -i /hello-world.tar

# You could also use a compose here if you desire
docker run -d --name hw1 hello-world:latest
docker run -d --name hw2 hello-world:latest

echo "Ran images!"

for CONTAINER in "hw1" "hw2"; do
    docker container wait "${CONTAINER}" > /dev/null
    docker logs "${CONTAINER}" > "/tmp/resim/outputs/${CONTAINER}-log.txt"
done
