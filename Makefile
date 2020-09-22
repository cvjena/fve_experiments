install:
	pip install . --no-deps --upgrade

build_sdist:
	@python setup.py build sdist

deploy: build_sdist
	./deploy_latest.sh

test_deploy: build_sdist
	REPO=pypitest ./deploy_latest.sh

get_version:
	@python -c "import fve_layer; print('v{}'.format(fve_layer.__version__))"

run_tests:
	python tests/main.py