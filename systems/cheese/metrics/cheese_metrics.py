import uuid
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from resim.metrics.python.metrics_writer import ResimMetricsWriter
from resim.metrics.proto.validate_metrics_proto import validate_job_metrics
from resim.metrics.python.metrics import (
  ScalarMetric,
  SeriesMetricsData
)
from resim.metrics.python.metrics_utils import (
  Timestamp,
  DoubleFailureDefinition,
  HistogramBucket,
  MetricStatus,
  MetricImportance
)

def read_test_results():
  input_dir = Path("/tmp/resim/inputs/logs")
  #input_dir = Path("/workspaces/open-core")
  cheese_file = input_dir.joinpath("cheese_log.json")
  cheese = 0.
  with cheese_file.open() as f:
    cheese_data = json.loads(f.read()) 
    cheese = cheese_data['score']
  return cheese


def scalar_metric_demo(writer):
  value = read_test_results()
  status = MetricStatus.PASSED_METRIC_STATUS
  if value < 0.71:
    status = MetricStatus.FAIL_WARN_METRIC_STATUS
  if value < 0.59:
    status = MetricStatus.FAIL_BLOCK_METRIC_STATUS
  failure_def = DoubleFailureDefinition(fails_above=9.9, fails_below=0)
  (
    writer
    .add_scalar_metric("Overall KPI")
    .with_failure_definition(failure_def)
    .with_unit("Score out of 10")
    .with_value(value)
    .with_description("Top level performance metric for the robot")
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

def double_summary_metric_demo(writer):
  classnames = SeriesMetricsData(
    name='Obj classes',
    series=np.array(["Obj 00","Obj 01", "Obj 02", "Obj 03"]),
    unit='')
  classspeeds = SeriesMetricsData(
    name='Max speed',
    series=np.array([0.85, 1.71, 0.99, 1.32]),
    unit='m/s',
    index_data=classnames)
  status_data = SeriesMetricsData(
    name="statuses_2",
    series=np.array([MetricStatus.PASSED_METRIC_STATUS for a in classnames.series]),
    unit='',
    index_data=classnames)
  (
    writer
    .add_double_summary_metric("Object speeds")
    .with_description("perception of object max speeds")
    .with_blocking(False)
    .with_should_display(True)
    .with_status(MetricStatus.PASSED_METRIC_STATUS)
    .with_value_data(classspeeds)
    .with_status_data(status_data)
    .with_importance(MetricImportance.HIGH_IMPORTANCE)
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
  time_data = SeriesMetricsData(
    name="times",
    series=np.array([Timestamp(secs=int(np.floor(a)), nanos=int(np.floor((a%1)*1E9))) for a in float_second]),
    unit='s')
  status_data = SeriesMetricsData(
    name="statuses_4",
    series=np.array([MetricStatus.PASSED_METRIC_STATUS for a in time_data.series]),
    unit='',
    index_data=time_data)
  value_data = SeriesMetricsData(
    name="states",
    series=np.array(50*["stationary"]+150*["moving"]+40*["stationary"]),
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

def write_proto(writer):
  metrics_proto = writer.write()
  validate_job_metrics(metrics_proto.metrics_msg)
  with open('/tmp/resim/outputs/metrics.binproto', 'wb') as f:
    f.write(metrics_proto.metrics_msg.SerializeToString())

def early_exit_if_batch_metrics():
    if Path("/tmp/resim/inputs/batch_metrics_config.json").is_file():
        print("Batch metrics not yet supported!")
        sys.exit(0)

if __name__ == "__main__":
  early_exit_if_batch_metrics()
  metrics_writer = ResimMetricsWriter(uuid.uuid4()) # Make metrics writer!
  scalar_metric_demo(metrics_writer)
  line_plot_metric_demo(metrics_writer)
  bar_chart_metric_demo(metrics_writer)
  histogram_metric_demo(metrics_writer)
  double_summary_metric_demo(metrics_writer)
  line_plot_metric_demo_2(metrics_writer)
  write_proto(metrics_writer)
