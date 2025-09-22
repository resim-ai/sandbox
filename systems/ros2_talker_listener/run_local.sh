#!/usr/bin/env bash
set -euo pipefail

cd $(dirname "${BASH_SOURCE[0]}")

docker compose up --abort-on-container-exit
docker compose down
