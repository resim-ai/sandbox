#!/bin/python

import argparse
import logging

import docker
import yaml
from docker.client import DockerClient

from scripts.utils import (
    get_client,
    get_project,
    register_experience_build,
    register_metrics_build,
    get_systems,
    get_branches,
    docker_ecr_auth,
    parse_version,
)

from scripts.builds_config import Builds, ImageRegistry, MetricsBuild, ExperienceBuild

logger = logging.getLogger("create_builds")
logger.setLevel(logging.INFO)


def list_command(builds, args):
    logger.info("Experience Builds:")
    for build in builds.experience_build_configs:
        logger.info("    %s", build)

    logger.info("Metrics Builds:")
    for build in builds.metrics_build_configs:
        logger.info("    %s", build)


def build_image(
    build: ExperienceBuild | MetricsBuild,
    registries: dict[str, ImageRegistry],
    docker_client: DockerClient,
) -> str:
    repo = build.repo
    registry = registries[repo.registry]
    docker_ecr_auth(docker_client, registry)
    command_path = build.build_command.path
    full_repo_name = (
        f"{registry.account_id}.dkr.ecr.{registry.region}.amazonaws.com/{repo.name}"
    )
    version = parse_version(build.version)
    tag = f"{build.version_tag_prefix}{version}"
    uri = f"{full_repo_name}:{tag}"
    response = docker_client.api.build(path=command_path, tag=uri, decode=True)
    for line in response:
        logger.info(" ".join((str(v) for v in line.values())))

    return uri


def push_image(uri: str, docker_client: DockerClient):
    response = docker_client.api.push(uri, stream=True, decode=True)
    for line in response:
        logger.info(" ".join((str(v) for v in line.values())))


def build_push(builds, args, *, push: bool):
    client = get_client(builds.resim_app_config)
    project_id = get_project(builds.project, client).project_id
    systems = get_systems(client, project_id)
    branches = get_branches(client, project_id)

    docker_client = docker.from_env()

    combined_map = builds.experience_build_configs | builds.metrics_build_configs
    for target in args.target_builds:
        if target not in builds.experience_build_configs:
            continue
        build = combined_map[target]
        uri = build_image(build, builds.registries, docker_client)

        if not push:
            continue

        push_image(uri, docker_client)
        register_experience_build(client, project_id, build, uri, systems, branches)

    for target in args.target_builds:
        if target not in builds.metrics_build_configs:
            continue

        build = combined_map[target]
        uri = build_image(build, builds.registries, docker_client)

        if not push:
            continue

        push_image(uri, docker_client)
        register_metrics_build(client, project_id, build, uri, systems)


def push_command(builds, args):
    build_push(builds, args, push=True)


def build_command(builds, args):
    build_push(builds, args, push=False)


def main():
    logging.basicConfig()
    parser = argparse.ArgumentParser(
        prog="create_builds",
        description="A simple CLI for building, pushing, and registering builds.",
    )
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List all resources")
    list_parser.set_defaults(func=list_command)

    # Build command
    build_parser = subparsers.add_parser("build", help="Build a selected build")
    build_parser.add_argument("target_builds", nargs="*")
    build_parser.set_defaults(func=build_command)

    # Push command
    push_parser = subparsers.add_parser("push", help="Push a selected build")
    push_parser.add_argument("target_builds", nargs="*")
    push_parser.set_defaults(func=push_command)

    args = parser.parse_args()

    with open("builds.yaml", "r", encoding="utf-8") as f:
        builds = Builds(**yaml.load(f, Loader=yaml.SafeLoader))
    args.func(builds, args)  # Call the appropriate function


if __name__ == "__main__":
    main()
