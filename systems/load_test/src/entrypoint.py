
import time
import shutil
import subprocess
import psutil
import uuid
import numpy as np
import threading

from resim.metrics.proto.validate_metrics_proto import validate_job_metrics
from resim.metrics.python.metrics_writer import ResimMetricsWriter

from resim.metrics.python.metrics import (
  Timestamp,
  DoubleFailureDefinition,
  SeriesMetricsData,
  HistogramBucket,
  MetricStatus,
  MetricImportance
)


NUM_PROCS = 8
MEM_GB = 24
B_PER_GB = 2**30
DURATION_S = 300
NUMBER_OF_METRICS = 30


def copy_input_to_output():
     shutil.copytree("/tmp/resim/inputs", "/tmp/resim/outputs/first_copy", dirs_exist_ok=True)
     shutil.copytree("/tmp/resim/inputs", "/tmp/resim/outputs/second_copy", dirs_exist_ok=True)


def double_summary_metric_demo(writer, idx):
    metrics_data = SeriesMetricsData(name = f"value_data_{idx}", series = np.array([42.0]))
    status_data = SeriesMetricsData(name = f"value_statuses_{idx}", series = np.array([MetricStatus.PASSED_METRIC_STATUS]))
    failure_def = DoubleFailureDefinition(fails_below=None, fails_above=None)
    (writer.add_double_summary_metric(name = f"Value Summary {idx}")
        .with_description("A random value")
        .with_blocking(False)
        .with_should_display(True)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_value_data(metrics_data)
        .with_status_data(status_data)
        .with_failure_definition(failure_def))

def write_proto(writer):
  metrics_proto = writer.write()
  validate_job_metrics(metrics_proto.metrics_msg)
  # Known location where the runner looks for metrics
  with open('/tmp/resim/outputs/metrics.binproto', 'wb') as f:
    f.write(metrics_proto.metrics_msg.SerializeToString())


def write_metrics():
    metrics_writer = ResimMetricsWriter(uuid.uuid4()) # Make metrics writer!
    for ii in range(NUMBER_OF_METRICS):
        double_summary_metric_demo(metrics_writer, ii)
    write_proto(metrics_writer)

def main():
    copythread = threading.Thread(target=copy_input_to_output)
    copythread.start()
    metricsthread = threading.Thread(target=write_metrics)
    metricsthread.start()

    procs = []
    for i in range(NUM_PROCS):
        mem_per_proc = MEM_GB // NUM_PROCS
        procs.append(subprocess.Popen(["/memory_load", f"{mem_per_proc}"]))

    done = False
    def print_status():
        while not done:
            print(f"CPU: {psutil.cpu_percent(percpu=True)} | Mem usage: {psutil.virtual_memory().used / B_PER_GB}GB")
            time.sleep(1)

    status_printer = threading.Thread(target=print_status)
    status_printer.start()

    time.sleep(DURATION_S)

    done = True
    for p in procs:
        p.kill()

    copythread.join()
    metricsthread.join()

if __name__ == "__main__":
    main()
