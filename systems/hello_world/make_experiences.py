#!/bin/python3

import os
import uuid
import json
import tempfile
import pathlib
import boto3
import typing
import subprocess
import datetime

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

arguments = [{
    "arguments": [3.14159265758979323, 0.5, 2.0],
    "num_iterations": ii
} for ii in range(1, 11)]


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
        "description": "Launch demo test experience.",
        "location": s3_path,
        "name": f"Launch demo test experience {id}",
        "creationTimestamp" : datetime.datetime.now().isoformat(),
        "experienceID" : id
    })
    response = create_experience.sync(client=resim_api_client,
                                      body=experience,
                                      project_id=project_id)
    assert response is not None


print("Make sure we have the CLI...")
subprocess.call(
    ["../../scripts/maybe_install_cli.sh"],
    cwd=pathlib.Path(__file__).parent)

staging_dir = tempfile.TemporaryDirectory()
staging_path = pathlib.Path(staging_dir.name)
for experience_contents in arguments:
    experience_id = str(uuid.uuid4())
    experience_path = staging_path / experience_id
    experience_path.mkdir()
    with open(experience_path / 'experience.json', 'w') as f:
        json.dump(experience_contents, f)
    destination = s3_prefix + experience_id + "/"

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
