
# mnist tmnist ctmnist
DATASET=${DATASET:-"ctmnist"}
EXTEND_SIZE=${EXTEND_SIZE:-128}
SCALE_DOWN=${SCALE_DOWN:-1}
N_JOBS=${N_JOBS:-3}

OPTS="${OPTS} --n_jobs ${N_JOBS}"
OPTS="${OPTS} --extend_size ${EXTEND_SIZE}"

if [[ $SCALE_DOWN == 1 ]]; then
	OPTS="${OPTS} --scale_down"
fi
