#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/launch_paper_protocol_tmux.sh" P3 1337 "$@"
