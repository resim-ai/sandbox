from pathlib import Path

BATCH_METRICS_CONFIG_PATH = Path("/tmp/resim/inputs/batch_metrics_config.json")

def run_batch_metrics():
    print("Running Batch Metrics")
    
    print("Finished Running Batch Metrics")
    
    
def run_test_metrics():
    print("Running the Precision Recall Calculation")
    
    print("Finished Running Test Metrics")

def main():
    """Entry point for the metrics builder script."""
    print("Starting to build metrics...")
    if BATCH_METRICS_CONFIG_PATH.exists():
        print("Running batch metrics...")
        run_batch_metrics()
    else:
        print("Running test metrics...")
        run_test_metrics()
        
if __name__ == "__main__":
    main() 