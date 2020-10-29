
GPU=${GPU:-0}
MODEL_TYPE=${MODEL_TYPE:-cv2_resnet50}
INPUT_SIZE=${INPUT_SIZE:-448}
PARTS_INPUT_SIZE=${PARTS_INPUT_SIZE:-224}

case $MODEL_TYPE in
	"inception" | "inception_imagenet" | "inception_inat" )
		PARTS_INPUT_SIZE=299
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=299
		elif [[ ${BIG:-0} == -1 ]]; then
			INPUT_SIZE=107
		else
			INPUT_SIZE=427
		fi
		;;
	"resnet" )
		PARTS_INPUT_SIZE=224
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=224
		else
			INPUT_SIZE=448
		fi
		;;
	"efficientnet" )
		PARTS_INPUT_SIZE=380
		INPUT_SIZE=380
		;;
esac


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
OPTS="${OPTS} --separate_model"
OPTS="${OPTS} --input_size ${INPUT_SIZE}"
OPTS="${OPTS} --parts_input_size ${PARTS_INPUT_SIZE}"
OPTS="${OPTS} --load_strict"
