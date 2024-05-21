#!/bin/python3

import os
import argparse
import json
import pathlib
from google.protobuf import text_format
import resim.experiences.proto.experience_pb2 as Experience

def apply_params(inputs_dir: pathlib.Path):
    params_path = inputs_dir / "parameters.json")
    if not params_path.is_file():
        return
    with open(params_path, "r") as f:
        params = json.load(f)
    if "velocity_cost" not in params:
        return

    with open(inputs_dir / "experience.sim", "r") as expfile:
        exp = text_format.Parse(expfile.read(), Experience.Experience())
    for movement_model in exp.dynamic_behavior.storyboard.movement_models:
        if movement_model.HasField("ilqr_drone"):
            movement_model.ilqr_drone.velocity_cost = eval(params["velocity_cost"])

    with open(inputs_dir / "experience.sim", "w") as expfile:
        expfile.write(str(exp))


def main():
    parser = argparse.ArgumentParser(
        prog = "apply_params",
        description = "Apply swept parameters if necessary.")
    parser.add_argument("--inputs_dir")
    args = parser.parse_args()
    assert args.inputs_dir is not None
    apply_params(pathlib.Path(Eargs.inputs_dir))


if __name__ == '__main__':
    main()
