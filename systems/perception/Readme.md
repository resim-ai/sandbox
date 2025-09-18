# Perception Case Study

This document will provide an example of how to use the ReSim Cloud Platform to test a robotic system's Perception module. The example that
we will talk about in this document is specifically the Object Detection Module of a Self Driving Car's Perception system. Check the [blog](https://www.resim.ai/blog/perception-case-study-analyzing-object-detection-systems-using-resim) for a detailed explanation.

## Experience Build

The experience stage is actually running the model ([OWL Vit model](https://huggingface.co/docs/transformers/en/model_doc/owlvit)) on the dataset. We have a sample of 10 images from the [Audi A2D2 dataset](https://www.a2d2.audi/en/dataset/) in the `experience/samples` directory, which are released under CC BY-ND 4.0 license. To build the experience docker container with the entrypoint required to run on the cloud, use

    cd experience
    ./build_experience.sh

We have a larger dataset on S3. Contact ReSim to access the folder or take any dataset with the annotations in a format similar to the `samples` directory and place it in both `experience/dataset` and `metrics/dataset` folder.  To run the experience with the sample dataset, run 

    ./run_experience.sh

This creates an `experience/output` folder with a `detections.csv` which will be used in the metrics stage

## Metrics Build

The metrics stage is responsible for going over the detections and analyzing and computing the Precision Recall Curve and other metrics. The dataset is similar to that of the experience stage. To build the metrics image, run 

    cd metrics
    ./build_metrics.sh

Metrics run in two ways. The first is individual test level metrics, and the second is the overall batch level metrics. 

### Test Metrics

To run the test metrics container locally, run

    ./run_metrics.sh

This creates a `metrics.binproto`, which can be uploaded to the [metrics debugger page](https://app.resim.ai/metrics-debugger) to view how the metrics would look like.

### Batch Metrics

To run batch metrics that aggregate results across multiple test runs, you need to configure the batch metrics settings:

1. **Configure batch_metrics_config.json**:
   ```json
   {
       "authToken": "your_resim_auth_token",
       "apiURL": "https://api.resim.ai/v1",
       "batchID": "your_batch_id",
       "projectID": "your_project_id"
   }
   ```

2. **Run batch metrics**:
   ```bash
   ./run_batch_metrics.sh
   ```

This will:
- Fetch metrics from all tests in the specified batch
- Calculate overall precision-recall curves across all tests
- Generate aggregated metrics including total true positives, false positives, and false negatives
- Create a comprehensive `metrics.binproto` with batch-level insights

**Note : To have batch metrics to be run locally, we need the batch to run once on the cloud**
