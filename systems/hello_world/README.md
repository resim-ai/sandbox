
# Hello, world!

## Cloud Running

This system is a very basic system that performs a simple computation (the sine
of a few specified arguments using a Taylor series) and produces some demo
metrics using a separate metrics build.

Here's how you can run it:

First, you need to set the following environment variables:

```
RESIM_SANDBOX_ECR="<aws-acct-id>.dkr.ecr.<region>.amazonaws.com"
export RESIM_SANDBOX_ECR_REPO="my-repo"
export RESIM_SANDBOX_ECR_REGION="<region>"
export RESIM_SANDBOX_PROJECT="SandboxProject"
export RESIM_SANDBOX_SYSTEM="SandboxSystem"
export RESIM_SANDBOX_BUILD_VERSION=$(git --rev-parse HEAD)
export RESIM_API_URL="https://api.resim.ai/v1/"
export RESIM_AUTH_URL="https://resim.us.auth0.com/"

export RESIM_SANDBOX_CLIENT_ID="<your-client-id>"
export RESIM_SANDBOX_S3_PREFIX="s3://<your-s3-bucket>/<experiences-path>"
```

Then, assuming you are authenticated with AWS so you can push to the given ECR
and s3 bucket, you should be able to run from this directory to create a new
build and a new metrics build in your project based on the scripts in
[src/entrypoint.py](./src/entrypoint.py) and
[metrics/metrics.py](./metrics/metrics.py) respectively.
```
./build_and_push.sh
```

Then, you can also create a set of experiences to run like so (assuming you have
python3 and pip installed): 

``` 
pip3 install -r requirements_lock.txt
./make_experiences.py
```

Now, you can kick off an experience run through the [ReSim
App](https://app.resim.ai). You should be able to navigate to the "Experiences"
tab on the left, select that experiences you want to run. Select the build and
metrics build you just made on the right side of the screen, and launch a batch
of sims with the "launch" button!

We encourage you to look through these scripts to check what they're doing and
play around with the build and the metrics build to see what you can run in the
app.


## Local Running

You can also run the build and metrics build locally using the following
sequence of commands:
```
./build.sh
./metrics/build.sh
./run_local.sh
```

This can be quite useful when editing and debugging the build and metrics
build. Note that these won't work from inside of a docker container since they
mount this directory to the build and metrics build images assuming it's on the
host.
