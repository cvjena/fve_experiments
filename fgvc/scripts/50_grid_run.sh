#!/usr/bin/env bash

CLUSTER=${CLUSTER:-0}

if [[ ${CLUSTER} != 0 ]]; then
	echo "=== CLUSTER execution enabled! ==="
	export OPTS=${OPTS:-"--no_progress"}

	NODE=${NODE:-s_mgx1,gpu_p100,gpu_v100}

	SBATCH_OPTS="${SBATCH_OPTS} --gres gpu:1"
	SBATCH_OPTS="${SBATCH_OPTS} -c 3"
	SBATCH_OPTS="${SBATCH_OPTS} --mem 32G"
	SBATCH_OPTS="${SBATCH_OPTS} -p ${NODE}"

	SBATCH_OUTPUT=${SBATCH_OUTPUT:-".sbatch/$(date +%Y-%m-%d_%H.%M.%S)"}
	mkdir -p $SBATCH_OUTPUT
	SBATCH_OPTS="${SBATCH_OPTS} --output ${SBATCH_OUTPUT}/fve_layer.%A.out"

	echo "slurm outputs will be saved under ${SBATCH_OUTPUT}"
fi

MODELS=${MODELS:-"inception inception_imagenet cv2_resnet50"}
DATASETS=${DATASETS:-"CUB200 NAB BIRDSNAP"}

export BATCH_SIZE=24
export FVE_TYPE=no
export PARTS=GLOBAL
export N_JOBS=4
export BIG=1

N_RUNS=${N_RUNS:-5}

for run in $(seq ${N_RUNS});
do
	for mt in $MODELS;
	do
		for ds in $DATASETS;
		do

			JOB_NAME=fve_${ds}_${mt}_#${run}

			if [[ ${CLUSTER} != 0 ]]; then
				SBATCH="sbatch --job-name ${JOB_NAME} ${SBATCH_OPTS}"
			fi

			OUTPUT_PREFIX=.results_GLOBAL \
			DATASET=$ds \
			MODEL_TYPE=$mt \
			${SBATCH} ./10_train.sh $@
		done
	done
done

if [[ ${CLUSTER} != 0 ]]; then
	rmdir --ignore-fail-on-non-empty ${SBATCH_OUTPUT}
fi
