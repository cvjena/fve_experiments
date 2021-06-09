
if [[ -z ${DATASET} ]]; then
	echo "DATASET variable is not set!"
	exit 1
fi

BATCH_SIZE=${BATCH_SIZE:-24}
UPDATE_SIZE=${UPDATE_SIZE:-64}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
OPTIMIZER=${OPTIMIZER:-adam}
EPOCHS=${EPOCHS:-60}
DEBUG=${DEBUG:-0}


if [[ ${DEBUG} != 0 ]]; then
	OPTS="${OPTS} --debug"
fi

# >>> LR definition >>>
INIT_LR=${INIT_LR:-1e-3}
LR_DECAY=${LR_DECAY:-1e-1}
LR_STEP=${LR_STEP:-1000}
LR_TARGET=${LR_TARGET:-1e-8}

LR=${LR:-"-lr ${INIT_LR} -lrd ${LR_DECAY} -lrs ${LR_STEP} -lrt ${LR_TARGET}"}
# >>>>>>>>>>>>>>>>>>>>>

# >>> Augmentations >>>
AUGMENTATIONS=${AUGMENTATIONS:-"random_crop random_flip color_jitter"}
OPTS="${OPTS} --augmentations ${AUGMENTATIONS}"
OPTS="${OPTS} --center_crop_on_val"
# >>>>>>>>>>>>>>>>>>>>>


OUTPUT_FOLDER=${OUTPUT_FOLDER:-".results"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"results"}
_now=$(date +%Y-%m-%d-%H.%M.%S.%N)
OUTPUT=${OUTPUT:-"${OUTPUT_FOLDER}/${OUTPUT_PREFIX}/${DATASET}/${OPTIMIZER}/${_now}"}

OPTS="${OPTS} --batch_size ${BATCH_SIZE}"
OPTS="${OPTS} --update_size ${UPDATE_SIZE}"
OPTS="${OPTS} --label_smoothing ${LABEL_SMOOTHING}"
OPTS="${OPTS} --optimizer ${OPTIMIZER}"
OPTS="${OPTS} --epochs ${EPOCHS}"
OPTS="${OPTS} --output ${OUTPUT}"
OPTS="${OPTS} --logfile ${OUTPUT}/output.log"
OPTS="${OPTS} ${LR}"
