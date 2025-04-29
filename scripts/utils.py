import base64
import boto3
import git
from docker.client import DockerClient
from resim.auth.python.device_code_client import DeviceCodeClient
from http import HTTPStatus

from resim.metrics.fetch_all_pages import fetch_all_pages
from resim_python_client.api.builds import create_build_for_branch
from resim_python_client.api.metrics_builds import create_metrics_build

from resim_python_client.client import AuthenticatedClient
from scripts.builds_config import ImageRegistry, ResimAppConfig, ExperienceBuild, MetricsBuild
from resim.metrics.fetch_all_pages import fetch_all_pages
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def parse_version(version: str) -> str:
    if version == "auto":
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    return version


def parse_branch(branch: str) -> str:
    if branch == "auto":
        repo = git.Repo(search_parent_directories=True)
        return repo.active_branch.name
    return branch


def docker_ecr_auth(client: DockerClient, registry: ImageRegistry):
    session = boto3.Session(profile_name=registry.auth.profile)
    ecr_client = session.client("ecr", region_name=registry.region)
    token = ecr_client.get_authorization_token()
    password = (
        base64.b64decode(token["authorizationData"][0]["authorizationToken"]).decode().split(":")[1]
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


def get_systems(client: AuthenticatedClient, project_id: str) -> dict[str, str]:
    system_pages = fetch_all_pages(list_systems.sync, client=client, project_id=project_id)
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
    branch = parse_branch(build.branch)
    version = parse_version(build.version)
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
    version = parse_version(build.version)
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
        assert response.status_code == HTTPStatus.CREATED, "Failed to add metrics build to system"
        logger.info("Added metrics build %s to %s system", metrics_build_id, system)
