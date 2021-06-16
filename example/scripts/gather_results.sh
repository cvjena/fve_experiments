#!/usr/bin/env bash

SACRED_CREDS_FILE=$(realpath "../sacred/config.sh")
source configs/00_python.sh
source configs/32_sacred.sh

$PYTHON gather_results.py $@
