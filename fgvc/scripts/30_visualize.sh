#!/usr/bin/env bash

source 00_common.sh
source configs/31_visualize.sh

if [[ -z ${DATA} ]]; then
	echo "DATA variable is not set!"
	exit 1
fi

if [[ -z ${DATASET} ]]; then
	echo "DATASET variable is not set!"
	exit 1
fi

if [[ -z ${PARTS} ]]; then
	echo "PARTS variable is not set!"
	exit 1
fi

$PYTHON run.py visualize \
	${DATA} ${DATASET} ${PARTS} \
	${OPTS} \
	$@
