#!/bin/python

import argparse
import base64
import logging
from http import HTTPStatus

import boto3
import docker
import git
import yaml
from docker.client import DockerClient
from pydantic import BaseModel
from resim.auth.python.device_code_client import DeviceCodeClient
from resim.metrics.fetch_all_pages import fetch_all_pages
from resim_python_client.api.builds import create_build_for_branch
from resim_python_client.api.metrics_builds import create_metrics_build
from resim_python_client.api.projects import (
    create_branch_for_project,
    list_branches_for_project,
    list_projects,
)
from resim_python_client.api.systems import add_system_to_metrics_build, list_systems
from resim_python_client.client import AuthenticatedClient
from resim_python_client.models import (
    BranchType,
    CreateBranchInput,
    CreateBuildForBranchInput,
    CreateMetricsBuildInput,
)

logger = logging.getLogger("create_builds")
logger.setLevel(logging.INFO)


class ResimAppConfig(BaseModel):
    client_id: str
    auth_url: str
    api_url: str


class RegistryAuth(BaseModel):
    profile: str


class ImageRegistry(BaseModel):
    account_id: str
    region: str
    auth: RegistryAuth


class ImageRepo(BaseModel):
    name: str
    registry: str


class BuildCommand(BaseModel):
    path: str


class ExperienceBuild(BaseModel):
    description: str
    repo: ImageRepo
    version_tag_prefix: str
    system: str
    branch: str
    version: str
    build_command: BuildCommand


class MetricsBuild(BaseModel):
    name: str
    repo: ImageRepo
    version_tag_prefix: str
    systems: list[str]
    version: str
    build_command: BuildCommand


class Builds(BaseModel):
    project: str
    registries: dict[str, ImageRegistry]
    resim_app_config: ResimAppConfig
    experience_build_configs: dict[str, ExperienceBuild]
    metrics_build_configs: dict[str, MetricsBuild]


def get_version(version: str) -> str:
    if version == "auto":
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    return version


def get_branch(branch: str) -> str:
    if branch == "auto":
        repo = git.Repo(search_parent_directories=True)
        return repo.active_branch.name
    return branch


def docker_ecr_auth(client: DockerClient, registry: ImageRegistry):
    session = boto3.Session(profile_name=registry.auth.profile)
    ecr_client = session.client("ecr", region_name=registry.region)
    token = ecr_client.get_authorization_token()
    password = (
        base64.b64decode(token["authorizationData"][0]["authorizationToken"])
        .decode()
        .split(":")[1]
    )
    registry_url = f"{registry.account_id}.dkr.ecr.{registry.region}.amazonaws.com"
    client.login(username="AWS", password=password, registry=registry_url)
    logger.info("Successfully authenticated to %s.", registry_url)


def get_client(config: ResimAppConfig) -> AuthenticatedClient:
    auth_client = DeviceCodeClient(domain=config.auth_url, client_id=config.client_id)
    token = auth_client.get_jwt()["access_token"]
    client = AuthenticatedClient(base_url=config.api_url, token=token)

    return client


def get_project(project: str, client: AuthenticatedClient) -> str:
    project_pages = fetch_all_pages(list_projects.sync, client=client)
    projects = {p.name: p for page in project_pages for p in page.projects}
    return projects[project]


def open_config() -> dict:
    with open("builds.yaml", "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def list_command(builds, args):
    logger.info("Experience Builds:")
    for build in builds.experience_build_configs:
        logger.info("    %s", build)

    logger.info("Metrics Builds:")
    for build in builds.metrics_build_configs:
        logger.info("    %s", build)


def get_systems(client: AuthenticatedClient, project_id: str) -> dict[str, str]:
    system_pages = fetch_all_pages(
        list_systems.sync, client=client, project_id=project_id
    )
    return {p.name: p.system_id for page in system_pages for p in page.systems}


def get_branches(client: AuthenticatedClient, project_id: str) -> dict[str, str]:
    branch_pages = fetch_all_pages(
        list_branches_for_project.sync, client=client, project_id=project_id
    )
    return {p.name: p.branch_id for page in branch_pages for p in page.branches}


def make_branch(client: AuthenticatedClient, project_id: str, branch: str):
    response = create_branch_for_project.sync(
        project_id=project_id,
        client=client,
        body=CreateBranchInput(branch_type=BranchType.CHANGE_REQUEST, name=branch),
    )
    assert response is not None, "Failed to make branch"
    logger.info("Registered branch with id %s", response.branch_id)

    return response.branch_id


def register_experience_build(
    client: AuthenticatedClient,
    project_id: str,
    build: ExperienceBuild,
    uri: str,
    systems: dict[str, str],
    branches: dict[str, str],
):
    branch = get_branch(build.branch)
    version = get_version(build.version)
    if branch not in branches:
        branches[branch] = make_branch(client, project_id, branch)

    response = create_build_for_branch.sync(
        project_id=project_id,
        branch_id=branches[branch],
        client=client,
        body=CreateBuildForBranchInput(
            image_uri=uri,
            system_id=systems[build.system],
            version=version,
            description=build.description,
        ),
    )
    assert response is not None
    logger.info("Registered experience build with id %s", response.build_id)


def register_metrics_build(
    client: AuthenticatedClient,
    project_id: str,
    build: MetricsBuild,
    uri: str,
    systems: dict[str, str],
):
    version = get_version(build.version)
    response = create_metrics_build.sync(
        project_id=project_id,
        client=client,
        body=CreateMetricsBuildInput(
            image_uri=uri,
            name=build.name,
            version=version,
        ),
    )
    assert response is not None
    logger.info("Registered metrics build with id %s", response.metrics_build_id)
    metrics_build_id = response.metrics_build_id

    for system in build.systems:
        response = add_system_to_metrics_build.sync_detailed(
            project_id=project_id,
            system_id=systems[system],
            metrics_build_id=metrics_build_id,
            client=client,
        )
        assert (
            response.status_code == HTTPStatus.CREATED
        ), "Failed to add metrics build to system"
        logger.info("Added metrics build %s to %s system", metrics_build_id, system)


def build_push(builds, args, *, push: bool):
    client = get_client(builds.resim_app_config)
    project_id = get_project(builds.project, client).project_id
    systems = get_systems(client, project_id)
    branches = get_branches(client, project_id)

    docker_client = docker.from_env()

    combined_map = builds.experience_build_configs | builds.metrics_build_configs
    for target in args.target_builds:
        build = combined_map[target]
        repo = build.repo
        registry = builds.registries[repo.registry]

        docker_ecr_auth(docker_client, registry)
        command_path = build.build_command.path

        full_repo_name = (
            f"{registry.account_id}.dkr.ecr.{registry.region}.amazonaws.com/{repo.name}"
        )
        version = get_version(build.version)
        tag = f"{build.version_tag_prefix}{version}"
        uri = f"{full_repo_name}:{tag}"

        response = docker_client.api.build(path=command_path, tag=uri, decode=True)
        for line in response:
            logger.info(" ".join((str(v) for v in line.values())))

        if not push:
            continue

        response = docker_client.api.push(
            repository=full_repo_name, tag=tag, stream=True, decode=True
        )
        for line in response:
            logger.info(" ".join((str(v) for v in line.values())))

        if isinstance(build, ExperienceBuild):
            register_experience_build(client, project_id, build, uri, systems, branches)
        elif isinstance(build, MetricsBuild):
            register_metrics_build(client, project_id, build, uri, systems)
        else:
            raise RuntimeError("Bad build type!")


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
    builds = Builds(**open_config())
    args.func(builds, args)  # Call the appropriate function


if __name__ == "__main__":
    main()
