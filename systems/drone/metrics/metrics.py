import asyncio
import mcap.reader
import uuid
import json
import collections
import numpy as np
import typing
import sys
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from google.protobuf import text_format

from batch_metrics import compute_batch_metrics

from mpl_toolkits.mplot3d.art3d import Line3DCollection

from resim.metrics.proto.validate_metrics_proto import validate_job_metrics
from resim.metrics.python.metrics_writer import ResimMetricsWriter

from resim.metrics.python.metrics import (
    Timestamp,
    DoubleFailureDefinition,
    ExternalFileMetricsData,
    SeriesMetricsData,
    GroupedMetricsData,
    HistogramBucket,
    MetricStatus,
    MetricImportance,
)

from decoder_factory import DecoderFactory
from resim.transforms.python import se3_python
from resim.transforms.python import so3_python

from resim.experiences.proto.experience_pb2 import Experience
from resim.experiences.proto.actor_pb2 import Actor


LOG_PATH = "/tmp/resim/inputs/logs/resim_log.mcap"
EXPERIENCE_PATH = "/tmp/resim/inputs/experience/experience.sim"
METRICS_PATH = "/tmp/resim/outputs/metrics.binproto"
BATCH_METRICS_CONFIG_PATH = Path("/tmp/resim/inputs/batch_metrics_config.json")


_TOPICS = ["/actor_states"]


def load_experience():
    with open(EXPERIENCE_PATH, "rb") as fp:
        experience = text_format.Parse(fp.read(), Experience())
    return experience


def update_wireframe(wireframe_collection: Line3DCollection, pose, geometry):
    wireframe = geometry.wireframe

    def mapped_point(index):
        return pose * np.array(wireframe.points[index].values)

    segments = [(mapped_point(e.start), mapped_point(e.end)) for e in wireframe.edges]
    wireframe_collection.set(segments=segments)


def load_log() -> list[dict[str, typing.Any]]:
    messages = collections.defaultdict(list)
    with open(LOG_PATH, "rb") as converted_log:
        reader = mcap.reader.make_reader(
            converted_log, decoder_factories=[DecoderFactory()]
        )
        for _, channel, _, message_proto in reader.iter_decoded_messages(
            topics=_TOPICS
        ):
            messages[channel.topic].append(message_proto)
    return messages


def make_gif_metric(
    writer, wireframe, poses: list[se3_python.SE3], times, goal
) -> None:
    # Data: 40 random walks as (num_steps, 3) arrays
    num_steps = 30
    trajectory = np.array([pose.translation() for pose in poses])

    # Attaching 3D axis to the figure
    plt.style.use("dark_background")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("ortho")
    ax.set_title("Ego Pose")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.plot(
        [trajectory[0, 0]],
        [trajectory[0, 1]],
        [trajectory[0, 2]],
        marker="o",
        color="red",
    )
    to_goal = ax.plot([goal[0]], [goal[1]], [goal[2]], linestyle="--", color="green")[0]

    # Create lines initially without data
    line = ax.plot([], [], [])[0]
    line_collection = ax.add_collection(Line3DCollection(segments=[]))

    # Creating the Animation object
    num_steps = len(poses)

    def animate(i: int):
        ax.set_title(f"Ego Pose: t {times[i]:.2f}s")
        line.set_data_3d(trajectory[: (i + 1), :].T)
        update_wireframe(line_collection, poses[i], wireframe)
        size = 7.5
        ax.set_xlim(trajectory[i, 0] - size, trajectory[i, 0] + size)
        ax.set_ylim(trajectory[i, 1] - size, trajectory[i, 1] + size)
        ax.set_zlim(trajectory[i, 2] - size, trajectory[i, 2] + size)
        to_goal.set_data_3d(*([trajectory[i, j], goal[j]] for j in range(3)))

    ani = animation.FuncAnimation(fig, animate, num_steps)
    pillow_writer = animation.PillowWriter(
        fps=10, metadata=dict(artist="Me"), bitrate=1800
    )
    ani.save("/tmp/resim/outputs/pose.gif", writer=pillow_writer)

    status = MetricStatus.PASSED_METRIC_STATUS

    data = ExternalFileMetricsData(name="pose_gif_data", filename="pose.gif")
    (
        writer.add_image_metric(name="pose_gif")
        .with_description("Ego Pose")
        .with_blocking(False)
        .with_should_display(True)
        .with_status(status)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_image_data(data)
    )

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.set_xlim(-750, 750)
    ax.set_ylim(-750, 750)
    ax.set_title("Top-down Ego Position")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    line = ax.plot([], [])[0]
    marker = ax.plot([trajectory[0, 0]], [trajectory[0, 1]], marker="x")[0]
    ax.plot([trajectory[0, 0]], [trajectory[0, 1]], marker="o", color="red")
    ax.plot([goal[0]], [goal[1]], marker="o", color="green")

    def animate_map(i: int):
        ax.set_title(f"Top-down Ego Position: t = {times[i]:.2f}s")
        line.set_data(trajectory[: (i + 1), :2].T)
        marker.set_data([trajectory[i, 0]], [trajectory[i, 1]])

    ani = animation.FuncAnimation(fig, animate_map, num_steps)
    pillow_writer = animation.PillowWriter(
        fps=10, metadata=dict(artist="Me"), bitrate=1800
    )
    ani.save("/tmp/resim/outputs/top_down.gif", writer=pillow_writer)

    data = ExternalFileMetricsData(name="top_down_gif_data", filename="top_down.gif")
    status = MetricStatus.PASSED_METRIC_STATUS
    (
        writer.add_image_metric(name="top_down_gif")
        .with_description("Top Down Pose")
        .with_blocking(False)
        .with_should_display(True)
        .with_status(status)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_image_data(data)
    )


def ego_metrics(writer, log):
    states_over_time = log["/actor_states"]

    id_to_states = collections.defaultdict(list)
    for bundle in log["/actor_states"]:
        for state in bundle.states:
            if state.is_spawned:
                id_to_states[state.id.data].append(state)

    experience = load_experience()
    ego_actors = [
        a
        for a in experience.dynamic_behavior.actors
        if a.actor_type == Actor.SYSTEM_UNDER_TEST
    ]
    ego_actor = ego_actors[0]
    ego_id = ego_actor.id.data
    ego_geometry = [
        g
        for g in experience.geometries
        if g.id.data == ego_actor.geometries[0].geometry_id.data
    ][0]

    ego_states = id_to_states[ego_id]
    ego_states = ego_states[0::10]

    def time_to_s(t):
        return t.seconds + 1e-9 * t.nanos

    poses = [se3_python.SE3.exp(s.state.ref_from_frame.algebra) for s in ego_states]
    times = np.array([time_to_s(s.time_of_validity) for s in ego_states])
    ego_movement_model = [
        m
        for m in experience.dynamic_behavior.storyboard.movement_models
        if m.actor_reference.id.data == ego_id
    ][0]
    ego_goal = np.array(ego_movement_model.ilqr_drone.goal_position)

    make_gif_metric(writer, ego_geometry, poses, times, ego_goal)

    ego_states = ego_states[0::10]

    times = np.array([time_to_s(s.time_of_validity) for s in ego_states])

    poses = [se3_python.SE3.exp(s.state.ref_from_frame.algebra) for s in ego_states]
    altitudes = np.array([p.translation()[-1] for p in poses])

    times = SeriesMetricsData("altitude_times", series=times, unit="s")
    altitudes = SeriesMetricsData("altitudes", series=altitudes, unit="m")
    statuses = SeriesMetricsData(
        "altitude statuses",
        series=np.array([MetricStatus.PASSED_METRIC_STATUS] * len(altitudes.series)),
    )

    (
        writer.add_line_plot_metric("Altitude over Time")
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

    velocities = [
        p.rotation() * np.array(s.state.d_ref_from_frame[3:])
        for p, s in zip(poses, ego_states)
    ]
    speeds = np.array([np.linalg.norm(v) for v in velocities])

    speeds = SeriesMetricsData("speeds", series=speeds, unit="m/s")
    statuses = SeriesMetricsData(
        "speed statuses",
        series=np.array([MetricStatus.PASSED_METRIC_STATUS] * len(speeds.series)),
    )

    (
        writer.add_line_plot_metric("Speed over Time")
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

    def status_from_speed(speed):
        status = MetricStatus.PASSED_METRIC_STATUS
        if speed > failure_def.fails_above:
            status = MetricStatus.FAIL_BLOCK_METRIC_STATUS
        elif speed > 20.0:
            status = MetricStatus.FAIL_WARN_METRIC_STATUS
        return status

    status = status_from_speed(max_speed)

    max_speed = GroupedMetricsData(
        "max_speed", category_to_series={"ego": np.array([max_speed])}, unit="m/s"
    )

    statuses = GroupedMetricsData(
        "max speed status", category_to_series={"ego": np.array([status])}
    )

    (
        writer.add_double_summary_metric(name="Max Speed")
        .with_description("Ego maximum speed")
        .with_blocking(False)
        .with_should_display(True)
        .with_status(status)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_value_data(max_speed)
        .with_status_data(statuses)
        .with_failure_definition(failure_def)
    )

    timestamps = GroupedMetricsData(
        "timestamps",
        category_to_series={
            "ego": np.array(
                [
                    Timestamp(
                        secs=s.time_of_validity.seconds, nanos=s.time_of_validity.nanos
                    )
                    for s in ego_states
                ]
            )
        },
    )
    speed_states = GroupedMetricsData(
        "speed_states",
        category_to_series={
            "ego": np.array([status_from_speed(s).name for s in speeds.series])
        },
        index_data=timestamps,
    )
    speed_states_status = GroupedMetricsData(
        "speed_states_status",
        category_to_series={
            "ego": np.array([status_from_speed(s) for s in speeds.series])
        },
        index_data=timestamps,
    )

    (
        writer.add_states_over_time_metric(name="Speed Status Over Time")
        .with_description("Ego maximum speed")
        .with_blocking(False)
        .with_should_display(True)
        .with_status(status)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .append_states_over_time_data(speed_states)
        .append_statuses_over_time_data(speed_states_status)
        .with_states_set({s.name for s in MetricStatus})
        .with_failure_states(
            {"FAIL_BLOCK_METRIC_STATUS", "FAIL_WARN_METRIC_STATUS", "NO_METRIC_STATUS"}
        )
    )

    mean_speed = np.mean(speeds.series)
    status = MetricStatus.PASSED_METRIC_STATUS
    failure_def = DoubleFailureDefinition(fails_above=None, fails_below=None)
    (
        writer.add_scalar_metric("Mean Ego Speed (m/s)")
        .with_failure_definition(failure_def)
        .with_value(mean_speed)
        .with_description("Mean ego speed during the sim.")
        .with_blocking(False)
        .with_should_display(True)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_status(status)
        .with_unit("m/s")
    )


def write_proto(writer):
    metrics_proto = writer.write()
    validate_job_metrics(metrics_proto.metrics_msg)
    # Known location where the runner looks for metrics
    with open(METRICS_PATH, "wb") as f:
        f.write(metrics_proto.metrics_msg.SerializeToString())


async def maybe_batch_metrics():
    if BATCH_METRICS_CONFIG_PATH.is_file():
        with open(
            BATCH_METRICS_CONFIG_PATH, "r", encoding="utf-8"
        ) as metrics_config_file:
            metrics_config = json.load(metrics_config_file)
        await compute_batch_metrics(
            token=metrics_config["authToken"],
            api_url=metrics_config["apiURL"],
            project_id=metrics_config["projectID"],
            batch_id=metrics_config["batchID"],
            metrics_path=METRICS_PATH,
        )

        sys.exit(0)


async def main():
    await maybe_batch_metrics()

    log = load_log()

    metrics_writer = ResimMetricsWriter(uuid.uuid4())  # Make metrics writer!
    ego_metrics(metrics_writer, log)
    write_proto(metrics_writer)

    # Event
    # Arrival. Time to arrive. Distance Travelled

    # States over time
    # Overspeed, warning

    # Bar chart
    # Histogram
    # Plotly


if __name__ == "__main__":
    asyncio.run(main())
