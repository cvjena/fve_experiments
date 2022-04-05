if [[ -z ${LOAD} ]]; then
	echo "LOAD variable is not set!"
	exit 1
fi

if [[ ${CENTER_CROP:-1} -eq 1 ]]; then
	OPTS="${OPTS} --center_crop_on_val"
fi

OPTS="${OPTS} --load ${LOAD}"
