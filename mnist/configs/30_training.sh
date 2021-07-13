
if [[ -z ${DATASET} ]]; then
	echo "DATASET variable is not set!"
	exit 1
fi

BATCH_SIZE=${BATCH_SIZE:-40}
UPDATE_SIZE=${UPDATE_SIZE:--1}
OPTIMIZER=${OPTIMIZER:-sgd}
EPOCHS=${EPOCHS:-80}
DEBUG=${DEBUG:-0}


if [[ ${DEBUG} != 0 ]]; then
	OPTS="${OPTS} --debug"
fi

# >>> LR definition >>>
INIT_LR=${INIT_LR:-1e-3}
LR_DECAY=${LR_DECAY:-1e-1}
LR_STEP=${LR_STEP:-30}
LR_TARGET=${LR_TARGET:-1e-8}

LR=${LR:-"-lr ${INIT_LR} -lrd ${LR_DECAY} -lrs ${LR_STEP} -lrt ${LR_TARGET}"}
# >>>>>>>>>>>>>>>>>>>>>


OUTPUT_PREFIX=${OUTPUT_PREFIX:-".results"}
OUTPUT=${OUTPUT:-"${OUTPUT_PREFIX}/${DATASET}/${OPTIMIZER}/$(date +%Y-%m-%d-%H.%M.%S.%N)"}

OPTS="${OPTS} --batch_size ${BATCH_SIZE}"
OPTS="${OPTS} --update_size ${UPDATE_SIZE}"
OPTS="${OPTS} --optimizer ${OPTIMIZER}"
OPTS="${OPTS} --epochs ${EPOCHS}"
OPTS="${OPTS} --output ${OUTPUT}"
OPTS="${OPTS} ${LR}"
