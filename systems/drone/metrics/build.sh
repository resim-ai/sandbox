#!/bin/bash

set -e

METRICS_DIR=$(dirname "$0")
cd "${METRICS_DIR}"

docker build -t drone_sim_metrics .
