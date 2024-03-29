FVE_TYPE=${FVE_TYPE:-no} # 'no', 'grad', or 'em'

N_COMPONENTS=${N_COMPONENTS:-1}
### COMP_SIZE
# < 0 -> pre-FVE is an 1x1 ConvLayer with #channels set to number of channels in the ConvMap
# = 0 -> pre-FVE is an identity (pre-FVE is disabled, default)
# > 0 -> pre-FVE is an 1x1 ConvLayer with #channels set to COMP_SIZE
COMP_SIZE=${COMP_SIZE:-0}

if [[ ${N_COMPONENTS} == 1 ]]; then
	# samples the µs uniformly from -1..1
	INIT_MU=${INIT_MU:-1}
else
	# samples the µs uniformly from -10..10
	INIT_MU=${INIT_MU:-${N_COMPONENTS}}
fi

INIT_SIG=${INIT_SIG:-1}
MASK_FEATURES=${MASK_FEATURES:-1}
NORMALIZE=${NORMALIZE:-0}
POST_FVE_SIZE=${POST_FVE_SIZE:-0}

# this aux_clf settings resulted in best results yet
AUX_LAMBDA=${AUX_LAMBDA:-0.5}
AUX_LAMBDA_RATE=${AUX_LAMBDA_RATE:-0.5}
AUX_LAMBDA_STEP=${AUX_LAMBDA_STEP:-20}

EMA_ALPHA=${EMA_ALPHA:-0.99}


_init_from_gap=${_init_from_gap:-0}
if [[ $_init_from_gap == 1 ]]; then

	echo "================================================"
	echo "================================================"
	echo "=== Initializing FVE parameters to match GAP ==="
	echo "================================================"
	echo "================================================"
	# this initializes the FVE-Layer params in the way
	# that the mu-part results in the same values as GAP

	COMP_SIZE=-1 # do not transform the features to a lower dimension

	INIT_MU=0
	INIT_SIG=1

	MASK_FEATURES=0
fi


OPTS="${OPTS} --fve_type ${FVE_TYPE}"
OPTS="${OPTS} --n_components ${N_COMPONENTS}"
OPTS="${OPTS} --comp_size ${COMP_SIZE}"
OPTS="${OPTS} --post_fve_size ${POST_FVE_SIZE}"
OPTS="${OPTS} --aux_lambda ${AUX_LAMBDA}"
OPTS="${OPTS} --aux_lambda_rate ${AUX_LAMBDA_RATE}"
OPTS="${OPTS} --aux_lambda_step ${AUX_LAMBDA_STEP}"
OPTS="${OPTS} --ema_alpha ${EMA_ALPHA}"
OPTS="${OPTS} --init_mu ${INIT_MU}"
OPTS="${OPTS} --init_sig ${INIT_SIG}"


if [[ ${MASK_FEATURES} == 1 ]]; then
	OPTS="${OPTS} --mask_features"
fi

if [[ ${NORMALIZE} == 1 ]]; then
	OPTS="${OPTS} --normalize"
fi
