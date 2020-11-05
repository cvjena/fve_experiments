#!/usr/bin/env bash

CLUSTER=${CLUSTER:-0}

if [[ ${CLUSTER} != 0 ]]; then
	echo "=== CLUSTER execution enabled! ==="
	export OPTS=${OPTS:-"--no_progress"}

	NODE=${NODE:-s_mgx1,gpu_p100,gpu_v100}

	SBATCH_OPTS="${SBATCH_OPTS} --gres gpu:1"
	SBATCH_OPTS="${SBATCH_OPTS} -c 3"
	SBATCH_OPTS="${SBATCH_OPTS} --mem 60G"
	SBATCH_OPTS="${SBATCH_OPTS} -p ${NODE}"

	SBATCH_OUTPUT=${SBATCH_OUTPUT:-".sbatch/$(date +%Y-%m-%d_%H.%M.%S)"}
	mkdir -p $SBATCH_OUTPUT
	SBATCH_OPTS="${SBATCH_OPTS} --output ${SBATCH_OUTPUT}/fve_layer.%A.out"

	echo "slurm outputs will be saved under ${SBATCH_OUTPUT}"
fi

PARTS=${PARTS:-"GLOBAL GT2 L1_pred"}
FVE=${FVE:-"no em grad"}
DATASETS=${DATASETS:-"CUB200"}

export BATCH_SIZE=12
export N_JOBS=4
export EMA_ALPHA=0.99

export MODEL_TYPE="cv2_resnet50"
export _init_from_gap=1

N_RUNS=${N_RUNS:-5}

for run in $(seq 1 ${N_RUNS});
do
	for pts in $PARTS;
	do
		for fve in $FVE;
		do
			for ds in $DATASETS;
			do

				JOB_NAME="fve(${fve})_parts${pts}_${ds}"

				if [[ ${CLUSTER} != 0 ]]; then
					SBATCH="sbatch --job-name ${JOB_NAME} ${SBATCH_OPTS}"
				fi

				OUTPUT_PREFIX=.results_ft_ResNet_parts \
				DATASET=$ds \
				PARTS=$pts \
				FVE_TYPE=$fve \
				${SBATCH} ./train.sh $@
			done
		done
	done
done

if [[ ${CLUSTER} != 0 ]]; then
	rmdir --ignore-fail-on-non-empty ${SBATCH_OUTPUT}
fi
