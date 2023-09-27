#!/usr/bin/env bash

export INIT_LR=${INIT_LR:-1e-3}
export BATCH_SIZE=${BATCH_SIZE:-32}
export OUTPUT_PREFIX=${OUTPUT_PREFIX:-"clf_only"}

./10_train.sh --only_clf \
	$@
