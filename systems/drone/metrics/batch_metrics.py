

import uuid

from resim.metrics.python.unpack_metrics import (
    unpack_metrics,
    UnpackedMetrics,
)
import numpy as np

from resim.metrics.python.metrics import (
  SeriesMetricsData,
  MetricImportance,    
  MetricStatus
)


from resim.metrics.proto.validate_metrics_proto import validate_job_metrics
from resim.metrics.python.metrics_writer import ResimMetricsWriter
from resim.metrics.proto.metrics_pb2 import JobMetrics
from resim_python_client.api.batches import list_jobs

from resim_python_client.client import AuthenticatedClient
from resim_python_client.api.batches import get_batch
from resim.metrics.fetch_all_pages import async_fetch_all_pages

from resim.metrics import fetch_job_metrics

async def compute_batch_metrics(
    *, token: str, api_url: str, project_id: str, batch_id: str, metrics_path: str
) -> None:
    client = AuthenticatedClient(base_url=api_url, token=token)

    batch = await get_batch.asyncio(project_id=project_id, batch_id=batch_id, client=client)

    job_pages = await async_fetch_all_pages(
        list_jobs.asyncio, client=client, project_id=project_id, batch_id=batch_id,
    )
    jobs = [e for page in job_pages for e in page.jobs]
    jobs = [
        fetch_job_metrics.JobInfo(
            job_id=uuid.UUID(job.job_id),
            batch_id=uuid.UUID(batch_id),
            project_id=uuid.UUID(project_id),
        )
        for job in jobs
    ]

    # metrics_protos, metrics_data_protos = fetch_job_metrics.fetch_job_metrics(
    #     token=token, base_url=api_url, jobs=jobs
    # )

    metrics_writer = ResimMetricsWriter(uuid.uuid4())  # Make metrics writer!
    print(batch)
    status = MetricStatus.PASSED_METRIC_STATUS
    barnames = SeriesMetricsData(
      name='Job Statuses',
      series=np.array(["PASSED", "WARNING", "BLOCKING"]),
      unit='')
    count = SeriesMetricsData(
      name='Job Status Counts',
      series=np.array([batch.job_metrics_status_counts.passed,
                       batch.job_metrics_status_counts.fail_warn,
                       batch.job_metrics_status_counts.fail_block
                       ], dtype=np.float64),
      unit='',
      index_data=barnames)
    status_data = SeriesMetricsData(
      name="statuses_1",
      series=np.array([MetricStatus.PASSED_METRIC_STATUS for a in count.series]),
      unit='',
      index_data=barnames)

    (metrics_writer.add_bar_chart_metric("job_statuses")
        .with_description("Job Statuses")
        .with_blocking(False)
        .with_should_display(True)
        .with_status(status)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .append_values_data(count, "Count")
        .append_statuses_data(status_data)
        .with_x_axis_name("Job Statuses")
        .with_y_axis_name("Count")
    )
     

    write_proto(metrics_writer, "/tmp/resim/outputs/metrics.binproto")

    
def write_proto(writer, metrics_path):
    metrics_proto = writer.write()
    validate_job_metrics(metrics_proto.metrics_msg)
    # Known location where the runner looks for metrics
    with open(metrics_path, 'wb') as f:
        f.write(metrics_proto.metrics_msg.SerializeToString())

    
