#!/usr/bin/env bash

export _parts=${PARTS:-L1_pred}
export _fve=${FVE_TYPE:-em}

export _batch_size=${BATCH_SIZE:-8}

export MODEL_TYPE=${MODEL_TYPE:-cv2_resnet50}
export DATASET=${DATASET:-CUB200}
export MASK_FEATURES=${MASK_FEATURES:-0}

export OUTPUT_PREFIX=${OUTPUT_PREFIX:-.pipeline}

_output="${OUTPUT_PREFIX}/${DATASET}/$(date +%Y-%m-%d-%H.%M.%S.%N)"

export VACUUM=0

#### 1. Fine-tune the model on the dataset and global parts ####

BATCH_SIZE=24 \
PARTS=GLOBAL \
FVE_TYPE=no \
OUTPUT="${_output}/global_ft" \
	./train.sh

#### 2. Warm-up the classifier and the FVE layer ####

_weights="${_output}/global_ft/clf_final.npz"
if [[ ! -f ${_weights} ]]; then
	echo "No model file (${_weights}) found! Did fine-tuning failed?"
	exit 1
fi

BATCH_SIZE=32 \
EPOCHS=25 \
INIT_LR=${INIT_LR:-1e-3} \
LR_STEP=${LR_STEP:-10} \
OUTPUT="${_output}/warmup" \
LOAD=${_weights} \
PARTS=${_parts} \
FVE_TYPE=${_fve} \
	./train.sh --only_clf --headless

#### 3. Fine-tune the model with parts and the FVE layer ####

_weights="${_output}/warmup/clf_final.npz"
if [[ ! -f ${_weights} ]]; then
	echo "No model file (${_weights}) found! Did warm-up failed?"
	exit 2
fi

BATCH_SIZE=${_batch_size} \
OUTPUT="${_output}/end2end_ft" \
LOAD=${_weights} \
PARTS=${_parts} \
FVE_TYPE=${_fve} \
	./train.sh
