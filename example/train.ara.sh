#!/usr/bin/env bash

NODE=${NODE:-s_mgx1,gpu_p100,gpu_v100}

N_RUNS=${N_RUNS:-1}
SBATCH=${SBATCH:-sbatch}
SBATCH_OPTS=${SBATCH_OPTS:-""}

OUTPUT_FNAME="fve_layer.%A.out"
if [[ $N_RUNS -gt 1 && $SBATCH == "sbatch" ]]; then
	SBATCH_OPTS="${SBATCH_OPTS} --array=1-${N_RUNS}"
	OUTPUT_FNAME="fve_layer.%A.%a.out"
fi

if [[ $NODE == "gpu_test" ]]; then
	JOB_NAME="FVELayer_testing"
else
	JOB_NAME=${JOB_NAME:"FVE(${FVE_TYPE})_${DATASET}_${PARTS}"}
fi


if [[ $SBATCH == "sbatch" ]]; then
	SBATCH_OUTPUT=${SBATCH_OUTPUT:-".sbatch/$(date +%Y-%m-%d_%H.%M.%S)"}
	mkdir -p $SBATCH_OUTPUT
	SBATCH_OPTS="${SBATCH_OPTS} --output ${SBATCH_OUTPUT}/${OUTPUT_FNAME}"
	echo "slurm outputs will be saved under ${SBATCH_OUTPUT}"

	SBATCH_OPTS="${SBATCH_OPTS} --job-name ${JOB_NAME}"
	export OPTS="--no_progress"
fi

SBATCH_OPTS="${SBATCH_OPTS} --gres gpu:1"
SBATCH_OPTS="${SBATCH_OPTS} -c 3"
SBATCH_OPTS="${SBATCH_OPTS} --mem 32G"
SBATCH_OPTS="${SBATCH_OPTS} -p ${NODE}"

$SBATCH $SBATCH_OPTS ./train.sh $@
