#!/usr/bin/env bash

export INIT_LR=${INIT_LR:-1e-3}
export BATCH_SIZE=${BATCH_SIZE:-32}

./train.sh --only_clf \
	$@
