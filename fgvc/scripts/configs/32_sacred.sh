
SACRED_CREDS_FILE=${SACRED_CREDS_FILE:-$(realpath ../sacred/config.sh)}

if [[ -f ${SACRED_CREDS_FILE} ]]; then
	echo "Sacred credentials found; sacred enabled."
	source ${SACRED_CREDS_FILE}
else
	echo "No sacred credentials found; disabling sacred."
	OPTS="${OPTS} --no_sacred"
fi
