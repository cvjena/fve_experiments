MODEL_TYPE=${MODEL_TYPE:-inception}

case $MODEL_TYPE in
	"inception" | "inception_imagenet" | "inception_inat" )
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=299
		elif [[ ${BIG:-0} == -1 ]]; then
			INPUT_SIZE=107
		else
			INPUT_SIZE=427
		fi
		;;
	"resnet" )
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=224
		else
			INPUT_SIZE=448
		fi
		;;
	"efficientnet" )
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


OPTS="${OPTS} --model_type ${MODEL_TYPE}"
OPTS="${OPTS} --model_type ${MODEL_TYPE}"
OPTS="${OPTS} --input_size ${INPUT_SIZE}"
OPTS="${OPTS} --load_strict"
