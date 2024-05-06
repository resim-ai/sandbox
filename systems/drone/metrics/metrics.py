

import mcap.reader
import uuid
import json
import collections
import numpy as np
import typing
import sys
from pathlib import Path
from dataclasses import dataclass

from resim.metrics.proto.validate_metrics_proto import validate_job_metrics
from resim.metrics.python.metrics_writer import ResimMetricsWriter

from resim.metrics.python.metrics import (
  Timestamp,
  DoubleFailureDefinition,
  SeriesMetricsData,
  GroupedMetricsData,
  HistogramBucket,
  MetricStatus,
  MetricImportance
)

from decoder_factory import DecoderFactory
from resim.transforms.python import se3_python
from resim.transforms.python import so3_python


EXPERIENCE_PATH = "/tmp/resim/inputs/experience/experience.json"
LOG_PATH = "/tmp/resim/inputs/logs/resim_log.mcap"


_TOPICS = ["/actor_states"]

def load_log() -> list[dict[str, typing.Any]]:
    messages = collections.defaultdict(list)
    with open(LOG_PATH, 'rb') as converted_log:
        reader = mcap.reader.make_reader(
            converted_log, decoder_factories=[DecoderFactory()])
        for _, channel, _, message_proto in reader.iter_decoded_messages(
                                                          topics=_TOPICS):
            messages[channel.topic].append(message_proto)
    return messages


def ego_metrics(writer, log):
    states_over_time = log["/actor_states"]

    id_to_states = collections.defaultdict(list)
    for bundle in log["/actor_states"]:
        for state in bundle.states:
            if state.is_spawned:
                id_to_states[str(state.id)].append(state)

    for _, state in id_to_states.items():
        ego_states = state

    time_to_s = lambda t: t.seconds + 1e-9 * t.nanos
    times = np.array([time_to_s(s.time_of_validity) for s in ego_states])

    poses = [se3_python.SE3.exp(s.state.ref_from_frame.algebra) for s in ego_states]
    altitudes = np.array([p.translation()[-1] for p in poses])


    times = SeriesMetricsData("altitude_times", series=times, unit='s')
    altitudes = SeriesMetricsData("altitudes", series=altitudes, unit='m')
    statuses = SeriesMetricsData(
        "altitude statuses",
        series=np.array([MetricStatus.PASSED_METRIC_STATUS] * len(altitudes.series)))
    
    (
      writer
      .add_line_plot_metric("Altitude over Time")
      .with_description("Ego altitude over time.")
      .with_blocking(False)
      .with_should_display(True)
      .with_status(MetricStatus.PASSED_METRIC_STATUS)
      .append_series_data(times, altitudes)
      .append_statuses_data(statuses)
      .with_importance(MetricImportance.HIGH_IMPORTANCE)
      .with_x_axis_name("Time")
      .with_y_axis_name("Altitude")
    )

    velocities = [p.rotation() * np.array(s.state.d_ref_from_frame[3:]) for p, s in zip(poses, ego_states)]
    speeds = np.array([np.linalg.norm(v) for v in velocities])
    
    speeds = SeriesMetricsData("speeds", series=speeds, unit='m/s')
    statuses = SeriesMetricsData(
        "speed statuses",
        series=np.array([MetricStatus.PASSED_METRIC_STATUS] * len(speeds.series)))
    
    (
      writer
      .add_line_plot_metric("Speed over Time")
      .with_description("Ego speed over time.")
      .with_blocking(False)
      .with_should_display(True)
      .with_status(MetricStatus.PASSED_METRIC_STATUS)
      .append_series_data(times, speeds)
      .append_statuses_data(statuses)
      .with_importance(MetricImportance.HIGH_IMPORTANCE)
      .with_x_axis_name("Time")
      .with_y_axis_name("Speed")
    )

    max_speed = np.max(speeds.series)
    failure_def = DoubleFailureDefinition(fails_below=None, fails_above=30.0)
    
    status = MetricStatus.PASSED_METRIC_STATUS
    if max_speed > failure_def.fails_above:
        status = MetricStatus.FAIL_BLOCK_METRIC_STATUS
        print(status)        
    elif max_speed > 20.0:
        status = MetricStatus.FAIL_WARN_METRIC_STATUS
        print(status)
    
    max_speed = GroupedMetricsData("max_speed", category_to_series={"ego":np.array([max_speed])}, unit="m/s")

    statuses = GroupedMetricsData("max speed status", category_to_series={"ego":np.array([status])})

    (writer.add_double_summary_metric(name = "Max Speed")
        .with_description("Ego maximum speed")
        .with_blocking(False)
        .with_should_display(True)
        .with_status(status)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_value_data(max_speed)
        .with_status_data(statuses)
        .with_failure_definition(failure_def))

    
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

  metrics_writer = ResimMetricsWriter(uuid.uuid4()) # Make metrics writer!
  ego_metrics(metrics_writer, log)
  bar_chart_metric_demo(metrics_writer)
  histogram_metric_demo(metrics_writer)
  line_plot_metric_demo_2(metrics_writer)
  states_over_time_metric_demo(metrics_writer)

  write_proto(metrics_writer)

if __name__ == "__main__":
    main()
