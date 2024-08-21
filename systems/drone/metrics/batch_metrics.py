import uuid

from collections import defaultdict

from resim.metrics.python.unpack_metrics import (
    unpack_metrics,
    UnpackedMetrics,
)
import numpy as np

from resim.metrics.python.metrics import (
    SeriesMetricsData,
    MetricImportance,
    MetricStatus,
    HistogramBucket,
)


import plotly.express as px
import pandas as pd
from resim.metrics.proto.validate_metrics_proto import validate_job_metrics
from resim.metrics.python.metrics_writer import ResimMetricsWriter
from resim.metrics.proto.metrics_pb2 import JobMetrics
from resim_python_client.api.batches import list_jobs

from resim_python_client.client import AuthenticatedClient
from resim_python_client.api.batches import get_batch
from resim.metrics.fetch_all_pages import async_fetch_all_pages

from resim.metrics import fetch_job_metrics
from resim.metrics.python.unpack_metrics import unpack_metrics


def compute_buckets(data: SeriesMetricsData):
    lower_bound = 0
    upper_bound = np.max(data.series)

    interval = upper_bound - lower_bound
    expansion_factor = 0.05

    upper_bound += expansion_factor * interval

    if upper_bound == lower_bound:
        # Every point is the same
        upper_bound += 1
        lower_bound -= 1

    bins = 20

    def boundary(i: int):
        frac = i / bins
        return lower_bound * (1 - frac) + upper_bound * frac

    return [
        HistogramBucket(lower=boundary(i), upper=boundary(i + 1)) for i in range(bins)
    ]


async def compute_batch_metrics(
    *, token: str, api_url: str, project_id: str, batch_id: str, metrics_path: str
) -> None:
    client = AuthenticatedClient(base_url=api_url, token=token)

    batch = await get_batch.asyncio(
        project_id=project_id, batch_id=batch_id, client=client
    )

    job_pages = await async_fetch_all_pages(
        list_jobs.asyncio,
        client=client,
        project_id=project_id,
        batch_id=batch_id,
    )
    jobs = [e for page in job_pages for e in page.jobs]
    job_infos = [
        fetch_job_metrics.JobInfo(
            job_id=uuid.UUID(job.job_id),
            batch_id=uuid.UUID(batch_id),
            project_id=uuid.UUID(project_id),
        )
        for job in jobs
    ]

    metric_protos, metrics_data_protos = fetch_job_metrics.fetch_job_metrics(
        token=token, base_url=api_url, jobs=job_infos
    )

    metrics_data = {}
    for job_id in metrics_data_protos:
        unpacked_metrics = unpack_metrics(
            metrics=[], metrics_data=metrics_data_protos[job_id], events=[]
        )
        metrics_data[job_id] = unpacked_metrics.metrics_data

    metrics_writer = ResimMetricsWriter(uuid.uuid4())  # Make metrics writer!
    print(batch)
    status = MetricStatus.PASSED_METRIC_STATUS
    barnames = SeriesMetricsData(
        name="Job Statuses", series=np.array(["PASSED", "WARNING", "BLOCKING"]), unit=""
    )
    count = SeriesMetricsData(
        name="Job Status Counts",
        series=np.array(
            [
                batch.job_metrics_status_counts.passed,
                batch.job_metrics_status_counts.fail_warn,
                batch.job_metrics_status_counts.fail_block,
            ],
            dtype=np.float64,
        ),
        unit="",
        index_data=barnames,
    )
    status_data = SeriesMetricsData(
        name="statuses_1",
        series=np.array([MetricStatus.PASSED_METRIC_STATUS for a in count.series]),
        unit="",
        index_data=barnames,
    )

    (
        metrics_writer.add_bar_chart_metric("job_statuses")
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

    metrics_data_by_name = defaultdict(dict)
    for job_id, datas in metrics_data.items():
        for data in datas:
            metrics_data_by_name[job_id][data.name] = data

    test_names = {uuid.UUID(job.job_id): job.experience_name for job in jobs}
    altitudes_df = pd.DataFrame(
        [
            [time, altitude, test_names[job_id]]
            for job_id in metrics_data
            for (time, altitude) in zip(
                metrics_data_by_name[job_id]["altitude_times"].series,
                metrics_data_by_name[job_id]["altitudes"].series,
            )
        ],
        columns=["time (s)", "altitude (m)", "test"],
    )

    fig = px.line(
        altitudes_df,
        x="time (s)",
        y="altitude (m)",
        color="test",
        title="layout.hovermode='closest'",
    )
    fig.update_layout(
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    (
        metrics_writer.add_plotly_metric("Altitudes")
        .with_description("Altitudes in all tests over time")
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_should_display(True)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_blocking(False)
        .with_plotly_data(fig.to_json())
    )

    speeds_df = pd.DataFrame(
        [
            [time, speed, test_names[job_id]]
            for job_id in metrics_data
            for (time, speed) in zip(
                metrics_data_by_name[job_id]["altitude_times"].series,
                metrics_data_by_name[job_id]["speeds"].series,
            )
        ],
        columns=["time (s)", "speed (m/s)", "test"],
    )

    fig = px.line(
        speeds_df,
        x="time (s)",
        y="speed (m/s)",
        color="test",
        title="layout.hovermode='closest'",
    )
    fig.update_layout(
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    (
        metrics_writer.add_plotly_metric("Speeds")
        .with_description("Speeds in all tests over time")
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_should_display(True)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_blocking(False)
        .with_plotly_data(fig.to_json())
    )

    allspeeds = SeriesMetricsData(
        name="all_speeds", series=np.array(speeds_df["speed (m/s)"])
    )
    allspeeds_statuses = SeriesMetricsData(
        name="all_speeds_statuses",
        series=np.array([MetricStatus.PASSED_METRIC_STATUS] * len(allspeeds.series)),
    )

    buckets = compute_buckets(allspeeds)

    (
        metrics_writer.add_histogram_metric(name=f"Drone Speed Distribution")
        .with_description(f"Drone speed distribution accross the batch")
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_should_display(True)
        .with_blocking(False)
        .with_values_data(allspeeds)
        .with_statuses_data(allspeeds_statuses)
        .with_buckets(buckets)
        .with_lower_bound(buckets[0].lower)
        .with_upper_bound(buckets[-1].upper)
        .with_x_axis_name("Speed (m/s)")
    )

    (
        metrics_writer.add_scalar_metric("mean_speed")
        .with_description("mean speed over the batch for longitudinal reporting")
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_should_display(False)  # Don't display. Only want for reports.
        .with_blocking(False)
        .with_value(np.mean(allspeeds.series))
    )

    write_proto(metrics_writer, "/tmp/resim/outputs/metrics.binproto")


def write_proto(writer, metrics_path):
    metrics_proto = writer.write()
    validate_job_metrics(metrics_proto.metrics_msg)
    # Known location where the runner looks for metrics
    with open(metrics_path, "wb") as f:
        f.write(metrics_proto.metrics_msg.SerializeToString())
