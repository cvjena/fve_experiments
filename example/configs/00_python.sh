_conda=${HOME}/.miniconda3
source ${_conda}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-chainer7}

if [[ $PROFILE == "1" ]]; then
    echo "Python profiler enabled!"
    PYTHON="python -m cProfile -o profile"

elif [[ ! -z $DRY_RUN ]]; then
    echo "Dry run enabled!"
	PYTHON="echo python"

else
    PYTHON="python"

fi

export OMP_NUM_THREADS=2
