
# Sandbox

[ReSim Documentation](https://docs.resim.ai/)

Welcome to [ReSim](https://www.resim.ai/)'s sandbox repository. This repository
is intended to demonstrate some of the capabilities of the ReSim platform and
how it can be leveraged to gain insights into the performance of your AI.

## Basics

Here's how ReSim works in a nutshell. Let's say that you have an AI system that
you would like to run a battery of tests on and collect metrics from those
tests. Here's a terse outline you would do that with ReSim. For the full
picture, please refer to our [documentation](https://docs.resim.ai/).

  1. Build and package your **test application** in a
     [Docker](https://docs.docker.com/get-started/overview/) image. This docker
     image should be able to run your test application based on a few input
     files that vary from test to test. In practice, these files will be made
     available by the execution framework using mounted volumes, so they do not
     need to be packaged in this image. Also, this image doesn't need to compute
     or collect metrics, but should ideally produce logs from which metrics can
     be computed.
  2. Push this docker image to a container registry to which the ReSim app has
     read access. Then you register that image with ReSim using our
     [cli](https://github.com/resim-ai/api-client). This results in the creation
     of what we call a **build**, which is a reference to your docker image as
     well as some metadata.
  3. Place each set of inputs (e.g. scenario description files) that you want to
     run your image with in a distinct directory in a cloud storage bucket to
     which the ReSim app has read access, and register each of these directories
     with the cli. This results in the creation of what we call an
     **experience** for each directory of inputs. The outputs/results of your
     test application for each experience will be available through the ReSim
     app once execution is complete. If any of these outputs is an MCAP file,
     then a link will be provided so you can instantly open that file in
     [Foxglove Studio](https://foxglove.dev/) for visualization.
  4. Once this is done, you can execute your test application **build** with one
     or more **experiences** through the ReSim app or using the cli.
  5. [Optional] Using the cli, you can also add a **metrics build** which
     references another Docker image which you've built and pushed. This image
     should be designed to read the logs output by the **build** and convert
     them to a structured format which the ReSim app recognizes uses to create
     plots and other visualizations.
	 

Once this is set up, our [Github action](https://github.com/resim-ai/action)
can be used to easily set up continuous integration testing so you get
continuous feedback on how your AI's performance is affected by every change to
your code.

<!-- TODO(michael) Add a picture of some job results with metrics here -->
	 
## This Repository

This respository is intended to demonstrate this basic workflow for a few simple
cases and can be used as a guide for those trying to set up their first ReSim
build and metrics build. These examples can be found in the `systems` directory
with one subdirectory per example.
