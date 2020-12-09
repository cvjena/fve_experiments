#!/usr/bin/env bash

OPTS=${OPTS:-""}

source configs/00_python.sh
source configs/10_dataset.sh
source configs/20_model.sh

$PYTHON ${SCRIPT_NAME} show \
	${DATASET}\
	${OPTS} \
	$@

