# Docker in Docker System

This directory is intended to demonstrate how one can package one or more docker
images into a single outer docker-in-docker-based image which can be used to run
multiple docker containers in a single ReSim test. Some notes on this:

 - The inner image is packed into the outer image as a tar in this
   example. There are definitely other, more-space-efficient ways to handle
   this in cases where there are multiple inner images that share layers.
   
 - The outer image/container does not pull any images at runtime. It simply
   loads them from tars. Even if we had the credentials set up to pull, doing so
   would likely very quickly exhaust rate limits when running in the ReSim
   app. Packing the needed data into the outer image allows ReSim to mirror all
   of the needed content and avoid this issue.
   
 - The outer image here expects mounts at `/tmp/resim/inputs` and
   `/tmp/resim/outputs` containing the experience inputs and a path for outputs
   to be placed respectively. This matches the behavior in the ReSim app. This
   matches the docs
   [here](https://docs.resim.ai/setup/build-images/#inputs-and-outputs).
   
In order to build and run a test run:

    1. Have `docker` installed and running. Make sure you've logged in so you
       can pull `hello-world:latest` and `docker:dind`.
	   
	2. Have `uuid-runtime` installed so you can run `uuidgen` to make ad-hoc
       directories on the host system for inputs/outputs.
	   
	3. Run ./build_and_test.sh
