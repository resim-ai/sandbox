from pathlib import Path
from test_metrics import run_test_metrics
from batch_metrics import run_batch_metrics
import uuid
from resim.metrics.python.metrics_writer import ResimMetricsWriter
from resim.metrics.proto.validate_metrics_proto import validate_job_metrics

BATCH_METRICS_CONFIG_PATH = Path("/tmp/resim/inputs/batch_metrics_config.json")
METRICS_PATH = "/tmp/resim/outputs/metrics.binproto"


def write_proto(writer):
  metrics_proto = writer.write()
  print("Validating Metrics")
  validate_job_metrics(metrics_proto.metrics_msg)
  
  print("Validated. Writing metrics proto file")
  # Known location where the runner looks for metrics
  with open(METRICS_PATH, 'wb') as f:
    f.write(metrics_proto.metrics_msg.SerializeToString())
  
  print("Metrics proto file written")    
    

def main():
    """Entry point for the metrics builder script."""
    writer = ResimMetricsWriter(uuid.uuid4())
    
    print("Starting to build metrics...")
    if BATCH_METRICS_CONFIG_PATH.exists():
        print("Running batch metrics...")
        run_batch_metrics()
    else:
        print("Running test metrics...")
        run_test_metrics(writer)
        write_proto(writer)
        
if __name__ == "__main__":
    main() 