_conda=${HOME}/.miniconda3
source ${_conda}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-chainer7cu11}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

if [[ $PROFILE == "1" ]]; then
    echo "Python profiler enabled!"
    PYTHON="python -m cProfile -o profile"


elif [[ $CUDA_MEMCHECK == "1" ]]; then
    PYTHON="/home/korsch/.miniconda3/bin/cuda-memcheck --save cuda-memcheck-$(date +%Y-%m-%d-%H.%M.%S.%N).out python"

elif [[ ${N_MPI:-0} -gt 1 ]]; then
	echo "=== MPI execution enabled! ==="

	OPTS="${OPTS} --mpi"

	N_MPI=${N_MPI:-2}
	HOSTFILE=${HOSTFILE:-hosts.conf}

	# create hosts file with localhost only
	if [[ ! -f ${HOSTFILE} ]]; then
		echo "localhost slots=${N_MPI}" > ${HOSTFILE}
	fi

	ENV="-x PATH -x OMP_NUM_THREADS -x DATA"
	ENV="${ENV} -x MONGODB_USER_NAME -x MONGODB_PASSWORD -x MONGODB_DB_NAME"
	ENV="${ENV} -x MONGODB_HOST -x MONGODB_PORT"

	PYTHON="orterun -n ${N_MPI} --hostfile ${HOSTFILE} --oversubscribe --bind-to none ${ENV} python"
else
    PYTHON="python"

fi

if [[ ! -z $DRY_RUN ]]; then
    echo "Dry run enabled!"
	PYTHON="echo ${PYTHON}"
fi

