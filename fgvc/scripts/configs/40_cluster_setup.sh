## here we create a port forward fo the Sacred DB
if [[ ${IS_CLUSTER:-0} == 1 ]]; then
	_user="korsch"
	_host="sigma25.inf-cv.uni-jena.de"

	socket_name=deep_$(hostname)
	# ensure, that the file does not exist!
	rm -f $socket_name

	TARGET="${_user}@${_host}"
	PORT=${MONGODB_PORT:-27017}
	# create master connection (with "-M")
	ssh -M -S $socket_name -fnNT -L localhost:$PORT:localhost:$PORT $TARGET

	# check the connection
	ssh -S $socket_name  -O check $TARGET
	code=$?
	if [[ $code != 0 ]]; then
		echo "Was not able to connect to ${TARGET}!"
		exit $code
	fi
	echo "Cluster execution enabled."
else
	echo "Non-Cluster execution enabled."
fi
