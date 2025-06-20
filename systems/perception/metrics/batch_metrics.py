import json
import uuid
from resim.metrics.fetch_job_metrics import fetch_job_metrics_by_batch
from resim.metrics.proto.validate_metrics_proto import validate_job_metrics
from resim.metrics.python.metrics import (MetricImportance, MetricStatus,
                                          ScalarMetric)
from resim.metrics.python.metrics_writer import ResimMetricsWriter


def run_batch_metrics():
    print("Writing Batch Metrics")
    
    try:
        # Read batch config
        with open("/tmp/resim/inputs/batch_metrics_config.json", "r", encoding="utf-8") as metrics_config_file:
            metrics_config = json.load(metrics_config_file)
            

        token = metrics_config["authToken"]
        api_url = metrics_config["apiURL"]
        batch_id = metrics_config["batchID"]
        project_id = metrics_config["projectID"]
        
        job_to_metrics = fetch_job_metrics_by_batch(
                token=token,
                api_url=api_url,
                project_id=project_id,
                batch_id=batch_id,
        )
        
        total_fp = 0
        total_tp = 0
        total_fn = 0

        # Go through the tests and add the stats
        for test_id, metrics_proto in job_to_metrics.items():
            try:
                for metric in metrics_proto.metrics:
                    if metric.name == "False Positives" and isinstance(metric, ScalarMetric):
                        total_fp += metric.value
                    elif metric.name == "True Positives" and isinstance(metric, ScalarMetric):
                        total_tp += metric.value
                    elif metric.name == "False Negatives" and isinstance(metric, ScalarMetric):
                        total_fn += metric.value
                        
            except Exception as e:
                    print(f"Error processing test {test_id}: {str(e)}")
                    continue
                
        metrics_writer = ResimMetricsWriter(uuid.uuid4())
        
        (
            metrics_writer.add_scalar_metric("Total False Positives")
            .with_description("Sum of False Positives in all experiences")
            .with_importance(MetricImportance.HIGH_IMPORTANCE)
            .with_value(total_fp)
            .with_unit("")
            .with_status(MetricStatus.PASSED_METRIC_STATUS)
        )
        
        (
            metrics_writer.add_scalar_metric("Total True Positives")
            .with_description("Sum of True Positives in all experiences")
            .with_importance(MetricImportance.HIGH_IMPORTANCE)
            .with_value(total_tp)
            .with_unit("")
            .with_status(MetricStatus.PASSED_METRIC_STATUS)
        )
        
        (
            metrics_writer.add_scalar_metric("Total False Negatives")
            .with_description("Sum of False Negatives in all experiences")
            .with_importance(MetricImportance.HIGH_IMPORTANCE)
            .with_value(total_fn)
            .with_unit("")
            .with_status(MetricStatus.PASSED_METRIC_STATUS)
        )
        
        # Add a summary text metric
        summary = f"""# Flight Batch Summary
            - Total False Positives: {total_fp}
            - Total True Positives: {total_tp} 
            - Total False Negatives: {total_fn}
            """
            
        (
            metrics_writer.add_text_metric("Batch Summary")
            .with_description("Summary of all Detection Sequences")
            .with_importance(MetricImportance.HIGH_IMPORTANCE)
            .with_status(MetricStatus.PASSED_METRIC_STATUS)
            .with_text(summary)
        )
        
        # Write and validate metrics
        metrics_proto = metrics_writer.write()
        validate_job_metrics(metrics_proto.metrics_msg)

        # Write to file
        output_path = Path("/tmp/resim/outputs/metrics.binproto")
        with output_path.open("wb") as metrics_out:
            metrics_out.write(metrics_proto.metrics_msg.SerializeToString())
        print(f"Batch metrics: Wrote metrics to {output_path}")
    
    except Exception as e:
        raise RuntimeError("Error processing batch metrics") from e
    
    print("Completed processing batch metrics. Exiting.")
    
    
