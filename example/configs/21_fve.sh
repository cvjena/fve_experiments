FVE_TYPE=${FVE_TYPE:-no} # 'no', 'grad', or 'em'

N_COMPONENTS=${N_COMPONENTS:-1}
COMP_SIZE=${COMP_SIZE:-256}
AUX_LAMBDA=${AUX_LAMBDA:-0.0}
EMA_ALPHA=${EMA_ALPHA:-0.99}

OPTS="${OPTS} --fve_type ${FVE_TYPE}"
OPTS="${OPTS} --n_components ${N_COMPONENTS}"
OPTS="${OPTS} --comp_size ${COMP_SIZE}"
OPTS="${OPTS} --aux_lambda ${AUX_LAMBDA}"
OPTS="${OPTS} --ema_alpha ${EMA_ALPHA}"
OPTS="${OPTS} --mask_features"
