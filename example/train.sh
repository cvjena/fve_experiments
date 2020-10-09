#!/usr/bin/env bash

OPTS=${OPTS:-""}

source configs/00_python.sh
source configs/10_dataset.sh
source configs/20_model.sh
source configs/21_fve.sh
source configs/30_training.sh

mkdir -p $OUTPUT
echo "Results are saved under ${OUTPUT}"

VACUUM=${VACUUM:-1}
if [[ $VACUUM == 1 ]]; then
	echo "=!=!=!= Removing folder ${OUTPUT} on error =!=!=!="
fi

$PYTHON run.py train \
	${DATA} ${DATASET} ${PARTS} \
	${OPTS} \
	$@

res=$?
if [[ ${res} != 0 && ${VACUUM} == 1 ]]; then
	echo "Error occured! Removing ${OUTPUT}"
	rm -r ${OUTPUT}
fi
