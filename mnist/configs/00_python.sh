_conda=${HOME}/.miniconda3
source ${_conda}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-DeepFVE}

SCRIPT_NAME=${SCRIPT_NAME:-main.py}

if [[ $PROFILE == "1" ]]; then
    echo "Python profiler enabled!"
    PYTHON="python -m cProfile -o profile"

elif [[ ! -z $DRY_RUN ]]; then
    echo "Dry run enabled!"
	PYTHON="echo python"

elif [[ $CUDA_MEMCHECK == "1" ]]; then
    PYTHON="/home/korsch/.miniconda3/bin/cuda-memcheck --save cuda-memcheck-$(date +%Y-%m-%d-%H.%M.%S.%N).out python"

else
    PYTHON="python"

fi

export OMP_NUM_THREADS=2
