

import os
import uuid
import shutil
import json
import tempfile
import pathlib
import numpy as np
import boto3
import typing
import subprocess
import numpy

from google.protobuf import text_format
import resim.experiences.proto.experience_pb2 as ex
from resim_auth_utils.device_code_client import DeviceCodeClient
from resim_python_client.client import AuthenticatedClient
from resim_python_client.api.experiences import create_experience
from resim_python_client.api.projects import list_projects
from resim_python_client.models.experience import Experience


project = os.getenv("RESIM_SANDBOX_PROJECT")
assert project is not None, "RESIM_SANDBOX_PROJECT must be set!"

api_url = os.getenv("RESIM_API_URL")
assert api_url is not None

auth_url = os.getenv("RESIM_AUTH_URL")
assert auth_url is not None

if auth_url.endswith("/"):
    auth_url = auth_url[:-1]

client_id = os.getenv("RESIM_SANDBOX_CLIENT_ID")
assert client_id is not None

LOCAL_EXPERIENCE_DIR = pathlib.Path(__file__).parent / "local_experience"

class HasNextPageToken(typing.Protocol):
    """A simple protocol for classes having the next_page_token field"""
    next_page_token: str

ResponseType = typing.TypeVar("ResponseType", bound=HasNextPageToken)

def fetch_all_pages(endpoint: typing.Callable[..., ResponseType],
                    *args: typing.Any,
                    **kwargs: typing.Any) -> list[ResponseType]:
    """
    Fetches all pages from a given endpoint.
    """
    responses = []
    responses.append(endpoint(*args, **kwargs))
    assert responses[-1] is not None

    page_token = responses[-1].next_page_token
    while page_token:
        responses.append(endpoint(*args, **kwargs, page_token=page_token))
        assert responses[-1] is not None
        page_token = responses[-1].next_page_token
    return responses



auth_client = DeviceCodeClient(domain=auth_url, client_id=client_id)
token = auth_client.get_jwt()["access_token"]

resim_api_client = AuthenticatedClient(base_url=api_url,
                                       token=token)


def get_project_id():
    project_id = None
    projects = fetch_all_pages(list_projects.sync, client=resim_api_client)
    projects = [p for page in projects for p in page.projects]
    return next(p for p in projects if p.name == project).project_id


project_id = get_project_id()


with open(LOCAL_EXPERIENCE_DIR / "experience.sim", "r") as  seed_file:
    seed_experience = text_format.Parse(seed_file.read(), ex.Experience())



# We create 50 experiences with a variety of goal positions
goal_positions = [np.random.uniform(low=-500.0, high=500.0, size=2) for _ in range(50)]
velocity_costs = [np.random.uniform(low=0.0, high=0.2) for _ in range(50)]


s3_prefix = os.getenv("RESIM_SANDBOX_S3_PREFIX")
assert s3_prefix is not None, "RESIM_SANDBOX_S3_PREFIX must be set!"

project = os.getenv("RESIM_SANDBOX_PROJECT")
assert project is not None, "RESIM_SANDBOX_PROJECT must be set!"


assert s3_prefix.startswith("s3://")
s3_bucket = s3_prefix.split("/")[2]
s3_key_prefix = "/".join(s3_prefix.split("/")[3:])


def pathlib_walk(path: pathlib.Path):
    """Pathlib equivalent of os.path.walk()"""
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    yield path, subdirs, [f for f in path.iterdir() if f.is_file()]
    for d in subdirs:
        yield from pathlib_walk(d)


def push_to_bucket(client, staging_path, path, bucketname, key_prefix):
    """Push the contents of of path to key_prefix / relpath(staging_path, path)"""
    for root, dirs, files in pathlib_walk(path):
        for f in files:
            key = key_prefix / f.relative_to(staging_path)
            client.upload_file(f, bucketname, str(key))


def register_experience(id, s3_path):
    """Register an experience at the given s3_path with ReSim"""
    experience = Experience.from_dict({
        "description": "Simple drone demo experience.",
        "location": s3_path,
        "name": f"Simple drone experience {id}",
    })
    response = create_experience.sync(client=resim_api_client,
                                      body=experience,
                                      project_id=project_id)
    assert response is not None


print("Make sure we have the CLI...")
subprocess.call(
    ["../../../scripts/maybe_install_cli.sh"],
    cwd=pathlib.Path(__file__).parent)

staging_dir = tempfile.TemporaryDirectory()
staging_path = pathlib.Path(staging_dir.name)
for velocity_cost, goal_position in zip(velocity_costs, goal_positions):
    experience_id = str(uuid.uuid4())
    experience_path = staging_path / experience_id
    experience_path.mkdir()

    experience = ex.Experience()
    experience.CopyFrom(seed_experience)

    for movement_model in experience.dynamic_behavior.storyboard.movement_models:
        if movement_model.HasField("ilqr_drone"):
            movement_model.ilqr_drone.goal_position[0:2] = goal_position
            movement_model.ilqr_drone.velocity_cost = velocity_cost

    with open(experience_path / 'experience.sim', 'w') as f:
        f.write(str(experience))
    destination = s3_prefix + experience_id + "/"

    shutil.copyfile(LOCAL_EXPERIENCE_DIR / "world.glb", experience_path / "world.glb")
    s3_client = boto3.client('s3')
    push_to_bucket(
        s3_client,
        staging_path,
        experience_path,
        s3_bucket,
        s3_key_prefix)
    s3_path = f"{s3_prefix}{experience_id}/"
    print(s3_path)
    register_experience(experience_id, s3_path)
