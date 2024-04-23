# Hello World :wave: :earth_africa:

This is an example build which, as you might guess, prints "Hello, World!" when
run! In addition to this, it will also list out any files it sees mounted in
`/tmp/resim/inputs`, which is the standard prefix for inputs when executing in
the ReSim app. The build is defined by the [Dockerfile](./Dockerfile) in this
directory. It can be built using [build.sh](./build.sh) and run locally using
[run_local.sh](./run_local.sh). If you want to run in re-run, you'll need to
[reach out](info@resim.ai) so we can get you set up. Once you've done so, you'll
need to define the following environment variables:

```
export RESIM_SANDBOX_ECR="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export RESIM_SANDBOX_ECR_REPO="?"
export RESIM_SANDBOX_ECR_REGION="${AWS_REGION}"
export RESIM_SANDBOX_PROJECT="?"
export RESIM_SANDBOX_SYSTEM="?"
```

And then you can run the [build_and_push.sh](./build_and_push.sh) script to
envbuild and push the `hello_world` image to your ECR and register it with ReSim.

Next, we need some experiences to run with. This build is so simple it can run
with any experience, but if you don't have any yet, you can set the
`RESIM_SANDBOX_S3_PREFIX` variable to a path in an s3 bucket for which the ReSim
app has read access
(e.g. `s3://my-experiences-bucket/my-sandbox-hello-experiences`). You'll also
need to set `RESIM_SANDBOX_PROJECT` as above. Then you can run
[make_experiences.sh](./make_experiences.sh) to create ten trivial experiences
to run your build with!

Now it's time to run your build! Go to the [ReSim App](https://app.resim.ai/)
and select the experiences page on the left sidebar. From here, find your
experiences, select them, and select your build from the panel on the righthand
side. Once you've selected both a build and at least one experience, the
"Launch" button should be active and you can click it to start your experience
running!
