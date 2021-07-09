#!/usr/bin/env bash

OPTS=${OPTS:-""}


source configs/00_python.sh
source configs/10_dataset.sh
source configs/20_model.sh
source configs/31_evaluate.sh

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

$PYTHON $MAIN_SCRIPT evaluate \
	${DATA} ${DATASET} ${PARTS} \
	${OPTS} \
	$@
