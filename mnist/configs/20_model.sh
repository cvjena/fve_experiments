
GPU=${GPU:-0}
MODEL_TYPE=${MODEL_TYPE:-resnet20_cifar10}
INPUT_SIZE=${INPUT_SIZE:-32}

LOAD=${LOAD:-""}
WEIGHTS=${WEIGHTS:-""}

if [[ ! -z ${LOAD} ]]; then
	if [[ ! -z ${WEIGHTS} ]]; then
		echo "Set either LOAD or WEIGHTS!"
		exit 1
	else
		OPTS="${OPTS} --load ${LOAD}"
	fi
elif [[ ! -z ${WEIGHTS} ]]; then
	if [[ ! -z ${LOAD} ]]; then
		echo "Set either LOAD or WEIGHTS!"
		exit 1
	else
		OPTS="${OPTS} --weights ${WEIGHTS}"
	fi
fi

OPTS="${OPTS} --gpu ${GPU}"
OPTS="${OPTS} --model_type ${MODEL_TYPE}"
OPTS="${OPTS} --input_size ${INPUT_SIZE}"
OPTS="${OPTS} --load_strict"
