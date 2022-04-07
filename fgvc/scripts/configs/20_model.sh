
GPU=${GPU:-0}

# MODEL_TYPE=${MODEL_TYPE:-chainercv2.resnet50}
MODEL_TYPE=${MODEL_TYPE:-cvmodelz.InceptionV3}
PREPARE_TYPE=${PREPARE_TYPE:-model}

INPUT_SIZE=${INPUT_SIZE:-299}
PARTS_INPUT_SIZE=${PARTS_INPUT_SIZE:-299}

# concat mean
FEATURE_AGG=${FEATURE_AGG:-concat}

if [[ ! -z $FP16 ]]; then
	export CHAINER_DTYPE=mixed16
fi

case $MODEL_TYPE in
	"cvmodelz.InceptionV3" | "cvmodelz.InceptionV3HD" | "chainercv2.inceptionv3" )
		PARTS_INPUT_SIZE=299
		PRE_TRAINING=${PRE_TRAINING:-inat}
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=299
		elif [[ ${BIG:-0} == -1 ]]; then
			INPUT_SIZE=107
			PARTS_INPUT_SIZE=107
		else
			INPUT_SIZE=427
		fi
		;;
	"cvmodelz.ResNet50" \
	| "cvmodelz.ResNet50HD" \
	| "chainercv2.resnet18" \
	| "chainercv2.resnet50" \
	| "chainercv2.resnet101" \
	)
		PARTS_INPUT_SIZE=224
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=224
		elif [[ ${BIG:-0} == -1 ]]; then
			INPUT_SIZE=112
			PARTS_INPUT_SIZE=112
		else
			INPUT_SIZE=448
		fi
		;;
	"chainercv2.efficientnet" )
		PARTS_INPUT_SIZE=380
		INPUT_SIZE=380
		;;
esac


LOAD=${LOAD:-""}
WEIGHTS=${WEIGHTS:-""}
PRE_TRAINING=${PRE_TRAINING:-imagenet}

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
OPTS="${OPTS} --prepare_type ${PREPARE_TYPE}"
OPTS="${OPTS} --pretrained_on ${PRE_TRAINING}"
OPTS="${OPTS} --input_size ${INPUT_SIZE}"
OPTS="${OPTS} --parts_input_size ${PARTS_INPUT_SIZE}"
OPTS="${OPTS} --feature_aggregation ${FEATURE_AGG}"
OPTS="${OPTS} --load_strict"
