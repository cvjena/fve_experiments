if [[ -z ${LOAD} ]]; then
	echo "LOAD variable is not set!"
	exit 1
fi

OPTS="${OPTS} --load ${LOAD}"
