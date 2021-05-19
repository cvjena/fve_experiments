#!/usr/bin/env bash
export CONDA_ENV=chainer7

NODE=${NODE:-s_mgx1,gpu_p100,gpu_v100}


N_RUNS=${N_RUNS:-1}
N_GPUS=${N_GPUS:-1}
SBATCH=${SBATCH:-sbatch}
SBATCH_OPTS=${SBATCH_OPTS:-""}

OUTPUT_FNAME="fve_layer.%A.out"
if [[ $N_RUNS -gt 1 && $SBATCH == "sbatch" ]]; then
	SBATCH_OPTS="${SBATCH_OPTS} --array=1-${N_RUNS}"
	OUTPUT_FNAME="fve_layer.%A.%a.out"
fi

JOB_NAME=${JOB_NAME:-"FVE(${FVE_TYPE})_${DATASET}_${PARTS}"}

if [[ $NODE == "gpu_test" ]]; then
	JOB_NAME="FVELayer_testing"
fi


if [[ $SBATCH == "sbatch" ]]; then
	SBATCH_OUTPUT=${SBATCH_OUTPUT:-".sbatch/$(date +%Y-%m-%d_%H.%M.%S)"}
	mkdir -p $SBATCH_OUTPUT
	SBATCH_OPTS="${SBATCH_OPTS} --output ${SBATCH_OUTPUT}/${OUTPUT_FNAME}"
	echo "slurm outputs will be saved under ${SBATCH_OUTPUT}"

	SBATCH_OPTS="${SBATCH_OPTS} --job-name ${JOB_NAME}"
	export OPTS="${OPTS} --no_progress"
fi


# values for a single GPU
CPUS=6
RAM=32
# this one defines the number of data loading processes per GPU
export N_JOBS=${N_JOBS:-$(( $CPUS - 1 ))}

if [[ $N_GPUS -gt 1 ]]; then
	CPUS=$(( $CPUS * $N_GPUS ))
	RAM=$((  $RAM  * $N_GPUS ))

	export MPI=1
	export N_MPI=${N_GPUS}
fi

SBATCH_OPTS="${SBATCH_OPTS} --gres gpu:${N_GPUS}"
SBATCH_OPTS="${SBATCH_OPTS} -c ${CPUS}"
SBATCH_OPTS="${SBATCH_OPTS} --mem ${RAM}G"
SBATCH_OPTS="${SBATCH_OPTS} -p ${NODE}"

$SBATCH $SBATCH_OPTS ./train.sh $@
