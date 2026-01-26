#!/usr/bin/env python3
"""
Script to generate Precision-Recall curve from detection_raw data.

This script shows how to use the raw detection data to calculate
precision and recall at different confidence thresholds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def calculate_precision_recall_curve(detection_data: pd.DataFrame, ground_truth_data: pd.DataFrame) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate precision-recall curve from detection and ground truth data.
    
    Args:
        detection_data: DataFrame with columns ['confidence', 'is_true_positive']
        ground_truth_data: DataFrame with columns ['is_detected']
    
    Returns:
        Tuple of (thresholds, precisions, recalls)
    """
    # Sort detections by confidence (descending)
    data = detection_data.sort_values('confidence', ascending=False)
    
    thresholds = []
    precisions = []
    recalls = []
    
    # Get total number of ground truth objects
    total_gt = len(ground_truth_data)
    
    # Calculate precision and recall at each confidence threshold
    for i in range(len(data)):
        threshold = data.iloc[i]['confidence']
        predictions_above_threshold = data.iloc[:i+1]
        
        tp = len(predictions_above_threshold[predictions_above_threshold['is_true_positive'] == True])
        fp = len(predictions_above_threshold[predictions_above_threshold['is_true_positive'] == False])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        
        thresholds.append(threshold)
        precisions.append(precision)
        recalls.append(recall)
    
    return thresholds, precisions, recalls

def plot_pr_curve(thresholds: List[float], precisions: List[float], recalls: List[float], 
                  title: str = "Precision-Recall Curve"):
    """Plot the precision-recall curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
    plt.scatter(recalls, precisions, c=thresholds, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Confidence Threshold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

def calculate_ap(precisions: List[float], recalls: List[float]) -> float:
    """Calculate Average Precision (AP) using the 11-point interpolation method."""
    # Convert to numpy arrays
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # 11-point interpolation
    recall_thresholds = np.linspace(0, 1, 11)
    interpolated_precisions = np.zeros_like(recall_thresholds)
    
    for i, r in enumerate(recall_thresholds):
        # Find precisions where recall >= r
        valid_precisions = precisions[recalls >= r]
        if len(valid_precisions) > 0:
            interpolated_precisions[i] = np.max(valid_precisions)
    
    return np.mean(interpolated_precisions)

# Example usage:
if __name__ == "__main__":
    # This would be replaced with actual data from resim metrics
    # For demonstration, create sample data
    np.random.seed(42)
    n_detections = 100
    n_gt_objects = 80
    
    # Simulate detection data
    sample_detections = pd.DataFrame({
        'confidence': np.random.uniform(0.1, 0.9, n_detections),
        'is_true_positive': np.random.choice([True, False], n_detections, p=[0.7, 0.3])
    })
    
    # Simulate ground truth data
    sample_gt = pd.DataFrame({
        'is_detected': np.random.choice([True, False], n_gt_objects, p=[0.6, 0.4])
    })
    
    print("Sample Detection Data:")
    print(sample_detections.head())
    print(f"\nTotal detections: {len(sample_detections)}")
    print(f"True positives: {sample_detections['is_true_positive'].sum()}")
    print(f"False positives: {(~sample_detections['is_true_positive']).sum()}")
    
    print(f"\nSample Ground Truth Data:")
    print(f"Total GT objects: {len(sample_gt)}")
    print(f"Detected GT objects: {sample_gt['is_detected'].sum()}")
    print(f"Missed GT objects (false negatives): {(~sample_gt['is_detected']).sum()}")
    
    # Calculate PR curve
    thresholds, precisions, recalls = calculate_precision_recall_curve(sample_detections, sample_gt)
    
    # Calculate Average Precision
    ap = calculate_ap(precisions, recalls)
    print(f"\nAverage Precision (AP): {ap:.3f}")
    
    # Plot the curve
    plot_pr_curve(thresholds, precisions, recalls, f"Sample PR Curve (AP = {ap:.3f})")
    
    print("\nTo use with real data:")
    print("1. Export detection_raw and ground_truth_raw data from resim metrics")
    print("2. Load both into pandas DataFrames")
    print("3. Call calculate_precision_recall_curve(detections_df, gt_df)")
    print("4. Plot with plot_pr_curve()")
