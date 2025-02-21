import uuid
import json
import numpy as np
import typing
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass

from resim.metrics.proto.validate_metrics_proto import validate_job_metrics
from resim.metrics.python.metrics_writer import ResimMetricsWriter

from resim.metrics.python.metrics import (
  ExternalFileMetricsData,
  Timestamp,
  DoubleFailureDefinition,
  SeriesMetricsData,
  GroupedMetricsData,
  HistogramBucket,
  MetricStatus,
  MetricImportance
)
 
@dataclass
class ExperienceConfig:
    arguments: float
    num_iterations: int

EXPERIENCE_PATH = "/tmp/resim/inputs/experience/experience.json"
LOG_PATH = "/tmp/resim/inputs/logs/log.json"

def load_experience() -> ExperienceConfig:
    with open(EXPERIENCE_PATH, "r") as f:
        config_json = json.load(f)
    return ExperienceConfig(**config_json)


def load_log() -> list[dict[str, typing.Any]]:
    with open(LOG_PATH, "r") as l:
        return json.load(l)


def double_summary_metric_demo(writer, experience, log):
    num_values = len(experience.arguments)
    assert len(log[-1]["partial_sums"]) == num_values
    metrics_data = (GroupedMetricsData(name = "final_value_data")
         .with_category_to_series({
            f"sin({experience.arguments[i]})": np.array([log[-1]["partial_sums"][i]]) for i in range(num_values)
         })
    )
    status_data = (GroupedMetricsData(name = "final_value_statuses")
         .with_category_to_series({
            f"sin({experience.arguments[i]})": np.array([MetricStatus.PASSED_METRIC_STATUS]) for i in range(num_values)
         })
    )
    failure_def = DoubleFailureDefinition(fails_below=None, fails_above=None)
    (writer.add_double_summary_metric(name = "Final Value Summary")
        .with_description("Final converged values for various arguments")
        .with_blocking(False)
        .with_should_display(True)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_value_data(metrics_data)
        .with_status_data(status_data)
        .with_failure_definition(failure_def))


def double_over_time_metric_demo(writer, experience, log):
    num_values = len(experience.arguments)
    assert len(log[-1]["partial_sums"]) == num_values

    metric = (writer.add_double_over_time_metric(name = "Convergence over time")
        .with_description("Error values over time for various arguments, omitted when zero.")
        .with_blocking(False)
        .with_should_display(True)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_y_axis_name("log10|error|")
        .with_failure_definitions([DoubleFailureDefinition(fails_below=None, fails_above=None)] * num_values))

    NANOS_PER_SEC = 1000000000
    def get_ts(t):
        return Timestamp(secs=t // NANOS_PER_SEC, nanos= t % NANOS_PER_SEC)
    time_series = SeriesMetricsData(name="times", series=np.array([get_ts(log[i]["time"]) for i in range(len(log))]))

    metric.with_start_time(time_series.series[0]).with_end_time(time_series.series[-1])

    for i in range(num_values):
        arg = experience.arguments[i]
        values = [np.fabs(log[j]["partial_sums"][i] - np.sin(arg)) for j in range(len(log))]
        values = [np.log10(x) if x != 0. else np.nan for x in values]
        value_series = SeriesMetricsData(name=f"dot_data_{arg}", series=np.array(values), index_data=time_series)
        status_series = SeriesMetricsData(name=f"dot_status_data_{arg}", series=np.array([MetricStatus.PASSED_METRIC_STATUS] * len(log)), index_data=time_series)

        metric.append_doubles_over_time_data(value_series, legend_series_name=f"arg={arg}")
        metric.append_statuses_over_time_data(status_series)

def scalar_metric_demo(writer, experience, log):
  num_values = len(experience.arguments)
  assert len(log[-1]["partial_sums"]) == num_values
  max_error_value = np.max([np.fabs(log[-1]["partial_sums"][i] - np.sin(experience.arguments[i])) for i in range(num_values)])
  status = MetricStatus.PASSED_METRIC_STATUS
  if max_error_value > 0.001:
    status = MetricStatus.FAIL_WARN_METRIC_STATUS
  if max_error_value > 0.1:
    status = MetricStatus.FAIL_BLOCK_METRIC_STATUS
  failure_def = DoubleFailureDefinition(fails_above=0.1, fails_below=0)
  (
    writer
    .add_scalar_metric("Max Error")
    .with_failure_definition(failure_def)
    .with_value(max_error_value)
    .with_description("Max error on final iteration accross values.")
    .with_blocking(False)
    .with_should_display(True)
    .with_importance(MetricImportance.HIGH_IMPORTANCE)
    .with_status(status)
  )


def line_plot_metric_demo(writer):
  time = SeriesMetricsData(
    name='Time',
    series=np.linspace(0.,60.,600),
    unit='s')
  accelx = SeriesMetricsData(
    name='roll',
    series=np.sin(time.series),
    unit='rad/s/s')
  accely = SeriesMetricsData(
    name='pitch',
    series=0.5*np.sin(time.series),
    unit='rad/s/s')
  accelz = SeriesMetricsData(
    name='yaw',
    series=0.25*np.cos(time.series),
    unit='rad/s/s')
  status_data = SeriesMetricsData(
    name="statuses",
    series=np.array([MetricStatus.PASSED_METRIC_STATUS for a in time.series]),
    unit='')
  (
    writer
    .add_line_plot_metric("Angular Acceleration")
    .with_description("Angular acceleration about pitch yaw roll axes.")
    .with_blocking(False)
    .with_should_display(True)
    .with_status(MetricStatus.PASSED_METRIC_STATUS)
    .append_series_data(time, accelx, "roll")
    .append_statuses_data(status_data)
    .append_series_data(time, accely, "pitch")
    .append_statuses_data(status_data)
    .append_series_data(time, accelz, "yaw")
    .append_statuses_data(status_data)
    .with_importance(MetricImportance.HIGH_IMPORTANCE)
    .with_x_axis_name("Time")
    .with_y_axis_name("Angular accel")
  )

def line_plot_metric_demo_2(writer):
  time = SeriesMetricsData(
    name='Time2',
    series=np.linspace(0.,60.,600),
    unit='s')
  shared_mem = SeriesMetricsData(
    name='shared memory usage',
    series=np.array([y+np.random.normal(1,0.05,1)[0] for y in np.tanh(np.array([x/30. -1. for x in time.series]))]),
    unit='GB')
  status_data = SeriesMetricsData(
    name="statuses0",
    series=np.array([MetricStatus.PASSED_METRIC_STATUS for a in time.series]),
    unit='')
  (
    writer
    .add_line_plot_metric("Shared Memory Usage")
    .with_description("Track the shared mem usage.")
    .with_blocking(False)
    .with_should_display(True)
    .with_status(MetricStatus.PASSED_METRIC_STATUS)
    .append_series_data(time, shared_mem, "shared memory usage.")
    .append_statuses_data(status_data)
    .with_importance(MetricImportance.HIGH_IMPORTANCE)
    .with_x_axis_name("Time")
    .with_y_axis_name("Memory used")
  )

def bar_chart_metric_demo(writer):
  barnames = SeriesMetricsData(
    name='Object classes',
    series=np.array(["Obj 00","Obj 01", "Obj 02", "Obj 03"]),
    unit='')
  precision = SeriesMetricsData(
    name='Precision',
    series=np.array([0.85, 0.71, 0.97, 0.99]),
    unit='',
    index_data=barnames)
  recall = SeriesMetricsData(
    name='Recall',
    series=np.array([0.93, 0.97, 0.79, 0.69]),
    unit='',
    index_data=barnames)
  status_data = SeriesMetricsData(
    name="statuses_1",
    series=np.array([MetricStatus.PASSED_METRIC_STATUS for a in precision.series]),
    unit='',
    index_data=barnames)
  (
    writer
    .add_bar_chart_metric("Object detection")
    .with_description("perception metrics on object classes")
    .with_blocking(False)
    .with_should_display(True)
    .with_status(MetricStatus.PASSED_METRIC_STATUS)
    .append_values_data(precision, "Precision")
    .append_statuses_data(status_data)
    .append_values_data(recall, "Recall")
    .append_statuses_data(status_data)
    .with_importance(MetricImportance.HIGH_IMPORTANCE)
    .with_x_axis_name("Object types")
    .with_y_axis_name("precision/recall")
  )

def histogram_metric_demo(writer):
  error = SeriesMetricsData(
    name='localization error',
    series=np.random.normal(loc=0.0, scale=.15, size=600),
    unit='m/s')
  status_data = SeriesMetricsData(
    name="statuses_3",
    series=np.array([MetricStatus.PASSED_METRIC_STATUS for a in error.series]),
    unit='')
  buckets = [
    HistogramBucket(-0.5, -0.3),
    HistogramBucket(-0.3, -0.1),
    HistogramBucket(-0.1, 0.1),
    HistogramBucket(0.1, 0.3),
    HistogramBucket(0.3, 0.5)
  ]
  (
    writer
    .add_histogram_metric("Localization error frequency")
    .with_description("Frequency over magnitude of localization error")
    .with_blocking(False)
    .with_should_display(True)
    .with_status(MetricStatus.PASSED_METRIC_STATUS)
    .with_values_data(error)
    .with_statuses_data(status_data)
    .with_buckets(buckets)
    .with_importance(MetricImportance.HIGH_IMPORTANCE)
    .with_lower_bound(-0.5)
    .with_upper_bound(0.5)
    .with_x_axis_name("localization error (m)")
  )

def states_over_time_metric_demo(writer):
  float_second = np.linspace(0., 60., 240) 
  time_data = GroupedMetricsData(
    name="sot_times",
    category_to_series={"main": np.array([Timestamp(secs=int(np.floor(a)), nanos=int(np.floor((a%1)*1E9))) for a in float_second])},
    unit='s')
  status_data = GroupedMetricsData(
    name="statuses_4",
    category_to_series={"main": np.array([MetricStatus.PASSED_METRIC_STATUS for a in time_data.category_to_series["main"]])},
    unit='',
    index_data=time_data)
  value_data = GroupedMetricsData(
    name="states",
    category_to_series={"main": np.array(50*["stationary"]+150*["moving"]+40*["stationary"])},
    unit='',
    index_data=time_data)
  states_set = {"stationary", "moving", "fault"}
  failure_states = {"fault"}
  series_name = "Operating mode"
  (
    writer
    .add_states_over_time_metric("Operating mode")
    .with_description("Operating modes over time")
    .with_blocking(False)
    .with_should_display(True)
    .with_status(MetricStatus.PASSED_METRIC_STATUS)
    .with_states_over_time_data([value_data])
    .with_statuses_over_time_data([status_data])
    .with_states_set(states_set)
    .with_failure_states(failure_states)
    .with_legend_series_names(["Op mode"])
    .with_importance(MetricImportance.HIGH_IMPORTANCE)
  )

def gif_metric_demo(writer):
  shutil.copy("/data/detection_clip_01.gif", "/tmp/resim/outputs/detection_clip_01.gif")

  METRIC_DATA = ExternalFileMetricsData(name="Clip", filename="detection_clip_01.gif")
  (
    writer
    .add_image_metric("Clip of interest")
    .with_description("A clip of intersection negotiation.")
    .with_blocking(False)
    .with_should_display(True)
    .with_status(MetricStatus.PASSED_METRIC_STATUS)
    .with_importance(MetricImportance.HIGH_IMPORTANCE)
    .with_image_data(METRIC_DATA)
  )
  


def write_proto(writer):
  metrics_proto = writer.write()
  validate_job_metrics(metrics_proto.metrics_msg)
  # Known location where the runner looks for metrics
  with open('/tmp/resim/outputs/metrics.binproto', 'wb') as f:
    f.write(metrics_proto.metrics_msg.SerializeToString())

def maybe_batch_metrics():
    if Path("/tmp/resim/inputs/batch_metrics_config.json").is_file():
        metrics_writer = ResimMetricsWriter(uuid.uuid4()) # Make metrics writer!
        line_plot_metric_demo(metrics_writer)
        bar_chart_metric_demo(metrics_writer)
        histogram_metric_demo(metrics_writer)
        line_plot_metric_demo_2(metrics_writer)
        states_over_time_metric_demo(metrics_writer)        
        write_proto(metrics_writer)
        sys.exit(0)

def main():
  maybe_batch_metrics()

  log = load_log()
  experience = load_experience()

  metrics_writer = ResimMetricsWriter(uuid.uuid4()) # Make metrics writer!
  gif_metric_demo(metrics_writer)
  double_summary_metric_demo(metrics_writer, experience, log)
  double_over_time_metric_demo(metrics_writer, experience, log)
  scalar_metric_demo(metrics_writer, experience, log)
  line_plot_metric_demo(metrics_writer)
  bar_chart_metric_demo(metrics_writer)
  histogram_metric_demo(metrics_writer)
  line_plot_metric_demo_2(metrics_writer)
  states_over_time_metric_demo(metrics_writer)

  write_proto(metrics_writer)

if __name__ == "__main__":
    main()
