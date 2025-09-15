import json
import uuid
from pathlib import Path
from resim.metrics.fetch_job_metrics import fetch_job_metrics_by_batch
from resim.metrics.proto.validate_metrics_proto import validate_job_metrics
from resim.metrics.python.metrics import (MetricImportance, MetricStatus,
                                          ScalarMetric, SeriesMetricsData)
from resim.metrics.python.metrics_writer import ResimMetricsWriter
from metric_charts import add_precision_recall_curve


def compute_precision_recall_curve(scores, tp_flags, fp_flags, total_ground_truth):
    """
    Compute precision and recall curve from sorted detection data.
    Matches the logic from test_metrics.py compute_pr_curve function.
    
    Args:
        scores: List of confidence scores (should be sorted in descending order)
        tp_flags: List of true positive flags (0 or 1) corresponding to each score
        fp_flags: List of false positive flags (0 or 1) corresponding to each score
        total_ground_truth: Total number of ground truth objects (TP + FN)
    
    Returns:
        tuple: (precision_curve, recall_curve) as lists of floats
    """
    if not scores or len(scores) != len(tp_flags) or len(scores) != len(fp_flags):
        raise ValueError("All input lists must have the same length and be non-empty")
    
    if total_ground_truth <= 0:
        raise ValueError("Total ground truth must be positive")
    
    # Convert to numpy arrays and use cumsum like the original function
    import numpy as np
    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(fp_flags)
    
    # Calculate precision and recall with epsilon to avoid division by zero (matching original)
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall = tp_cum / (total_ground_truth + 1e-8)
    
    return precision.tolist(), recall.tolist()


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
        
        # Collect all series data for overall precision-recall curve
        combined_data = []

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
                
                
            try:
                # Collect series data for this test
                test_scores = None
                test_tp = None
                test_fp = None
                
                for metric_data in metrics_proto.metrics_data:
                    if metric_data.name == "Detection Score Series" and isinstance(metric_data,SeriesMetricsData):
                        print(f"Size of Metrics Data for test: {test_id} =  {metric_data.series.shape}")
                        test_scores = metric_data.series.tolist()
                    elif metric_data.name == "True Positive Series" and isinstance(metric_data,SeriesMetricsData):
                        print(f"Size of True Positive Series for test: {test_id} =  {metric_data.series.shape}")
                        test_tp = metric_data.series.tolist()
                    elif metric_data.name == "False Positive Series" and isinstance(metric_data,SeriesMetricsData):
                        print(f"Size of False Positive Series for test: {test_id} =  {metric_data.series.shape}")
                        test_fp = metric_data.series.tolist()
                
                # Combine into one tuple
                if test_scores and test_tp and test_fp and len(test_scores) == len(test_tp) == len(test_fp):
                    for i in range(len(test_scores)):
                        combined_data.append((test_scores[i], test_tp[i], test_fp[i]))
                    print(f"Added {len(test_scores)} detections from test {test_id}")
                        
            except Exception as e:
                print(f"Error processing test {test_id}: {str(e)}")
                continue
        
        # Sort all combined data by confidence score (descending)
        if combined_data:
            print(f"\nSorting {len(combined_data)} total detections by confidence score...")
            sorted_data = sorted(combined_data, key=lambda x: x[0], reverse=True)
            
            # Extract the three sorted lists
            all_scores, all_tp, all_fp = zip(*sorted_data)
            
            print(f"Sorted data - Top 5 scores: {all_scores[:5]}")
            print(f"Corresponding TP flags: {all_tp[:5]}")
            print(f"Corresponding FP flags: {all_fp[:5]}")
            
            # Calculate overall precision-recall curve
            total_ground_truth = total_tp + total_fn
            print(f"\nCalculating precision-recall curve with {total_ground_truth} total ground truth objects...")
            
            precision_curve, recall_curve = compute_precision_recall_curve(
                list(all_scores), 
                list(all_tp), 
                list(all_fp), 
                total_ground_truth
            )
            
            print(f"Precision-Recall curve calculated with {len(precision_curve)} points")
            print(f"Final Precision: {precision_curve[-1]:.3f}, Final Recall: {recall_curve[-1]:.3f}")
            
            # Print some key points from the curve
            print(f"Top 5 precision values: {[f'{p:.3f}' for p in precision_curve[:5]]}")
            print(f"Top 5 recall values: {[f'{r:.3f}' for r in recall_curve[:5]]}")
            
        else:
            print("No series data found to combine")
            precision_curve = None
            recall_curve = None
                
        metrics_writer = ResimMetricsWriter(uuid.uuid4())
        
        # Add precision-recall curve if we have the data
        if precision_curve is not None and recall_curve is not None:
            print("Adding precision-recall curve to metrics...")
            add_precision_recall_curve(metrics_writer, precision_curve, recall_curve)
        
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
            .with_tag(key="RESIM_SUMMARY", value="1")
        )
        
        # Add a summary text metric
        summary = f"""- Total False Positives: {total_fp}
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
    
    
