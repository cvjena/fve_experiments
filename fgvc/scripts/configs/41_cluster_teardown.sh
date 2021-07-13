
if [[ ${IS_CLUSTER:-0} == 1 ]]; then
	# exit the connection
	ssh -S $socket_name -O exit $TARGET
fi
