#!/usr/bin/env bash
_conda=${HOME}/.miniconda3
source ${_conda}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-chainer7cu11}

PYTHON="python"

if [[ ${DRY_RUN:-0} == 1 ]]; then
	PYTHON="echo ${PYTHON}"
fi

GPU=${GPU:-0}
N_CLASSES=${N_CLASSES:-5}
EPOCHS=${EPOCHS:-100}
SAMPLES_PER_DIM=${SAMPLES_PER_DIM:-16}

_OPTS="${_OPTS} --no_plot"
_OPTS="${_OPTS} --device ${GPU}"
_OPTS="${_OPTS} --epochs ${EPOCHS}"
_OPTS="${_OPTS} --n_classes ${N_CLASSES}"

N_RUNS=${N_RUNS:-"5"}
COMPONENTS=${COMPONENT:-"1 5"}
DIMENSIONS=${DIMENSIONS:-"2 4 8 16 32 64 128 256"}

_now="$(date +%Y-%m-%d-%H.%M.%S.%N)"
OUTPUT_PREFIX=${OUTPUT_PREFIX:-".results"}
OUTPUT="${OUTPUT_PREFIX}/${_now}"
mkdir -p $OUTPUT

for run in $(seq ${N_RUNS}); do
	for n_dims in ${DIMENSIONS}; do
		for n_comps in ${COMPONENTS}; do

			echo "=== CONFIG (run #${run}): #comps=${n_comps} #dims=${n_dims} ==="

			OPTS="${_OPTS}"
			n_samples=$(($n_dims * $SAMPLES_PER_DIM))
			batch_size=$(($n_samples * $N_CLASSES))
			# echo $n_comps $n_dims $n_samples

			output=${OUTPUT}/${run}_${n_comps}_${n_dims}.json


			OPTS="${OPTS} --n_dims ${n_dims}"
			OPTS="${OPTS} --n_samples ${n_samples}"
			OPTS="${OPTS} --batch_size ${batch_size}"
			OPTS="${OPTS} --n_components ${n_comps}"
			OPTS="${OPTS} --output ${output}"

			$PYTHON main.py $OPTS $@
		done
	done
done | tee ${OUTPUT}/output.log


# remove output folder if it is empty
if [[ -d ${OUTPUT} ]]; then
	rmdir --ignore-fail-on-non-empty ${OUTPUT}
fi
