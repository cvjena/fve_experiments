#!/usr/bin/env bash
PYTHON=python

export OMP_NUM_THREADS=2

OPTS=${OPTS:-""}

######## Dataset options ########
DATA=${DATA:-"${HOME}/Data/info.yml"}
DATASET=${DATASET:-"CUB200"}
PARTS=${PARTS:-"GLOBAL"}

LABEL_SHIFT=${LABEL_SHIFT:-1}
N_JOBS=${N_JOBS:-3}

OPTS="${OPTS} --n_jobs ${N_JOBS}"
OPTS="${OPTS} --label_shift ${LABEL_SHIFT}"
OPTS="${OPTS} --swap_channels"
######## Model options ########
MODEL_TYPE=${MODEL_TYPE:-inception}

case $MODEL_TYPE in
	"inception" | "inception_imagenet" | "inception_inat" )
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=299
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

######## FVE options ########
FVE_TYPE=${FVE_TYPE:-em} # 'no', 'grad', or 'em'

N_COMPONENTS=${N_COMPONENTS:-1}
COMP_SIZE=${COMP_SIZE:-256}
LOSS_LAMBDA=${LOSS_LAMBDA:-0.9}

OPTS="${OPTS} --fve_type ${FVE_TYPE}"
OPTS="${OPTS} --n_components ${N_COMPONENTS}"
OPTS="${OPTS} --comp_size ${COMP_SIZE}"
OPTS="${OPTS} --loss_lambda ${LOSS_LAMBDA}"
OPTS="${OPTS} --mask_features"

######## Training options ########
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
UPDATE_SIZE=${UPDATE_SIZE:-64}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
OPTIMIZER=${OPTIMIZER:-rmsprop}
EPOCHS=${EPOCHS:-60}
DEBUG=${DEBUG:-0}


if [[ ${DEBUG} != 0 ]]; then
	OPTS="${OPTS} --debug"
fi

# >>> LR definition >>>
INIT_LR=${INIT_LR:-1e-4}
LR_DECAY=${LR_DECAY:-1e-1}
LR_STEP=${LR_STEP:-20}
LR_TARGET=${LR_TARGET:-1e-6}

LR=${LR:-"-lr ${INIT_LR} -lrd ${LR_DECAY} -lrs ${LR_STEP} -lrt ${LR_TARGET}"}
# >>>>>>>>>>>>>>>>>>>>>
OUTPUT=${OUTPUT:-".results/${DATASET}/${OPTIMIZER}/$(date +%Y-%m-%d-%H.%M.%S.%N)"}

OPTS="${OPTS} --gpu ${GPU}"
OPTS="${OPTS} --batch_size ${BATCH_SIZE}"
OPTS="${OPTS} --update_size ${UPDATE_SIZE}"
OPTS="${OPTS} --label_smoothing ${LABEL_SMOOTHING}"
OPTS="${OPTS} --optimizer ${OPTIMIZER}"
OPTS="${OPTS} --epochs ${EPOCHS}"
OPTS="${OPTS} --output ${OUTPUT}"
OPTS="${OPTS} ${LR}"

######## End of options ########
# echo $OPTS
# exit

mkdir -p $OUTPUT
echo "Results are saved under ${OUTPUT}"

VACUUM=${VACUUM:-1}
if [[ $VACUUM == 1 ]]; then
	echo "=!=!=!= Removing folder ${OUTPUT} on error =!=!=!="
fi

$PYTHON run.py \
	${DATA} ${DATASET} ${PARTS} \
	${OPTS} \
	$@

res=$?
if [[ ${res} != 0 && ${VACUUM} == 1 ]]; then
	echo "Error occured! Removing ${OUTPUT}"
	rm -r ${OUTPUT}
fi
