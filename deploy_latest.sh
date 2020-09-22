#!/usr/bin/env bash

current_version=$(python -c "import fve_layer; print(fve_layer.__version__)")

REPO=${REPO:-pypi}

echo "Uploading to ${REPO} ..."

twine upload \
	--repository ${REPO} \
	dist/*${current_version}.tar.gz \

ret_code=$?
if [[ $ret_code == 0 ]]; then
	echo "OK"
fi
