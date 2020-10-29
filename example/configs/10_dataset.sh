
DATA=${DATA:-"${HOME}/Data/info.yml"}
DATASET=${DATASET:-"CUB200"}
PARTS=${PARTS:-"GLOBAL"}

LABEL_SHIFT=${LABEL_SHIFT:-1}
N_JOBS=${N_JOBS:-3}

OPTS="${OPTS} --n_jobs ${N_JOBS}"
OPTS="${OPTS} --label_shift ${LABEL_SHIFT}"
# OPTS="${OPTS} --swap_channels"

if [[ $PARTS == "GLOBAL" ]]; then
	OPTS="${OPTS} --cache_images"
fi
