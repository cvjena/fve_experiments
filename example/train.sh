#!/usr/bin/env bash

OPTS=${OPTS:-"--no_snapshot"}

source configs/00_python.sh
source configs/10_dataset.sh
source configs/20_model.sh
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

if [[ -z ${DATASET} ]]; then
	echo "DATASET variable is not set!"
	exit 1
fi

if [[ -z ${PARTS} ]]; then
	echo "PARTS variable is not set!"
	exit 1
fi
{ # try
	$PYTHON main.py train \
		${DATA} ${DATASET} ${PARTS} \
		${OPTS} \
		$@
} || { # catch

	if [[ ${VACUUM} == 1 ]]; then
		echo "Error occured! Removing ${OUTPUT}"
		rm -r ${OUTPUT}
	fi
}


# remove output folder if it is empty
if [[ -d ${OUTPUT} ]]; then
	rmdir --ignore-fail-on-non-empty ${OUTPUT}
fi
