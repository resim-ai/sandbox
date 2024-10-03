import argparse

import logging
import os
import uuid
import shutil
import json
import tempfile
import pathlib
import numpy as np
import boto3
import typing
import numpy

from google.protobuf import text_format
import resim.experiences.proto.experience_pb2 as ex
from resim_auth_utils.device_code_client import DeviceCodeClient
from resim_python_client.client import AuthenticatedClient
from resim_python_client.api.experiences import create_experience
from resim_python_client.api.projects import list_projects
from resim_python_client.models import CreateExperienceInput
from resim_python_client.api.experience_tags import add_experience_tag_to_experience


LOCAL_EXPERIENCE_DIR = pathlib.Path(__file__).parent / "local_experience"

logger = logging.getLogger(__name__)


def register_experience(
    *, experience_id, client, s3_path, project_id, id_to_location_map, experience_tag_id
):
    """Register an experience at the given s3_path with ReSim"""
    experience = CreateExperienceInput(
        description="Simple drone demo experience.",
        location=s3_path,
        name=f"testing2 - Simple drone experience {experience_id}",
    )

    response = create_experience.sync(
        client=client, body=experience, project_id=project_id
    )
    logger.debug(response)
    id_to_location_map[response.experience_id] = s3_path

    if experience_tag_id:
        response = add_experience_tag_to_experience.sync_detailed(
            client=client,
            project_id=project_id,
            experience_id=response.experience_id,
            experience_tag_id=experience_tag_id,
        )
        logger.debug(response)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auth-config-file",
        help="JSON file describing the apiURL, authURL, and clientID to use.",
        required=True,
    )
    parser.add_argument(
        "--experience-s3-prefix",
        help="s3 prefix to push experiences to.",
        required=True,
    )
    parser.add_argument(
        "--project-id",
        help="project id for the project to add the experiences to.",
        required=True,
    )
    parser.add_argument(
        "--experience-tag-id",
        help="optional experience tag to add to all experiences created",
        default="",
    )

    args = parser.parse_args()

    with open(args.auth_config_file, "r", encoding="utf8") as fp:
        auth_config = json.load(fp)

    logger.info("Authenticating...")
    auth_client = DeviceCodeClient(
        domain=auth_config["authURL"], client_id=auth_config["clientID"]
    )
    token = auth_client.get_jwt()["access_token"]

    client = AuthenticatedClient(base_url=auth_config["apiURL"], token=token)

    with open(LOCAL_EXPERIENCE_DIR / "experience.sim", "r") as seed_file:
        seed_experience = text_format.Parse(seed_file.read(), ex.Experience())

    # We create 50 experiences with a variety of goal positions
    goal_positions = [
        np.append(
            np.random.uniform(low=-500.0, high=500.0, size=2),
            np.random.uniform(low=20.0, high=300.0),
        )
        for _ in range(200)
    ]
    velocity_costs = [np.random.uniform(low=0.0, high=0.2) for _ in range(200)]

    s3_prefix = args.experience_s3_prefix
    assert s3_prefix.startswith("s3://")
    s3_bucket = s3_prefix.split("/")[2]
    s3_key_prefix = "/".join(s3_prefix.split("/")[3:])

    s3 = boto3.client("s3")

    for velocity_cost, goal_position in zip(velocity_costs, goal_positions):
        experience_id = str(uuid.uuid4())

        experience = ex.Experience()
        experience.CopyFrom(seed_experience)

        for movement_model in experience.dynamic_behavior.storyboard.movement_models:
            if movement_model.HasField("ilqr_drone"):
                movement_model.ilqr_drone.goal_position[0:3] = goal_position
                movement_model.ilqr_drone.velocity_cost = velocity_cost

        destination = s3_prefix + experience_id + "/"
        logger.info("Writing experience to %s", destination)
        response = s3.put_object(
            Body=str(experience),
            Bucket=s3_bucket,
            Key=f"{s3_key_prefix}{experience_id}/experience.sim",
        )
        logger.debug(response)

        response = s3.upload_file(
            Filename=(LOCAL_EXPERIENCE_DIR / "world.glb"),
            Bucket=s3_bucket,
            Key=f"{s3_key_prefix}{experience_id}/world.glb",
        )
        logger.debug(response)

        id_to_location_map = {}
        register_experience(
            client=client,
            experience_id=experience_id,
            project_id=args.project_id,
            s3_path=destination,
            id_to_location_map=id_to_location_map,
            experience_tag_id=args.experience_tag_id,
        )

        with open("id_to_location_map.json", "w", encoding="utf8") as fp:
            json.dump(id_to_location_map, fp)


if __name__ == "__main__":
    main()
