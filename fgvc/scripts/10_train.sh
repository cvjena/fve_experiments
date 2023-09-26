#!/usr/bin/env bash

source 00_common.sh
source configs/21_fve.sh
source configs/30_training.sh
source configs/32_sacred.sh


mkdir -p $OUTPUT
echo "Results are saved under ${OUTPUT}"

VACUUM=${VACUUM:-1}
if [[ $VACUUM == 1 ]]; then
	echo "=!=!=!= On error, removing folder ${OUTPUT} =!=!=!="
fi

if [[ -z ${DATA} ]]; then
	echo "DATA variable is not set!"
	exit 1
fi

if [[ ! -f ${DATA} ]]; then
	echo "provided DATA file with dataset and model informations is missing!"
	echo "Please ../../example.yml to ${DATA} and adjust the variables accordingly!"
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
source configs/40_cluster_setup.sh

{ # try
	$PYTHON $MAIN_SCRIPT train \
		${DATA} ${DATASET} ${PARTS} \
		${OPTS} \
		$@
} || { # catch

	if [[ ${VACUUM} == 1 ]]; then
		echo "Error occured! Removing ${OUTPUT}"
		rm -r ${OUTPUT}
	fi
}

source configs/41_cluster_teardown.sh


# remove output folder if it is empty
if [[ -d ${OUTPUT} ]]; then
	rmdir --ignore-fail-on-non-empty ${OUTPUT}
fi
