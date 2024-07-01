
import numpy as np
import uuid
import sys
from collections import defaultdict

from resim_python_client.client import AuthenticatedClient
from resim_python_client.api.batches import list_jobs
from resim.metrics import fetch_job_metrics
from resim.metrics.python.unpack_metrics import (
    unpack_metrics,
    UnpackedMetrics,
)
from resim.metrics.python.metrics import (
    Metric,
    ScalarMetric,
)
from resim.metrics.python.metrics_utils import MetricStatus


from resim_python_client.api.batches import (
    list_metrics_for_job, list_metrics_data_for_job)

from resim_python_client.api.reports import (
    get_report)

from resim.metrics.python.metrics_writer import ResimMetricsWriter
from resim.metrics.proto.metrics_pb2 import JobMetrics

def run_report_metrics(*,
                      token: str,
                      api_url: str,
                      project_id: str,                      
                      report_id: str,
                      metrics_path: str) -> None:
    """Run the report metrics and save them to the given metrics_path."""

    # Fetch the jobs for this report
    client = AuthenticatedClient(
        base_url=api_url,
        token=token)
    report_response = get_report.sync(project_id, report_id, client=client)

    report = report_response.report
    report_name = report.name
    print(f"Fetched report!")
    # print(f"Fetched {len(metrics_data_protos)} metrics data protos!")

    # print("Writing report metrics binproto!")
    # with open(metrics_path, "wb") as metrics_file:
    #     metrics_file.write(metrics_msg.SerializeToString())