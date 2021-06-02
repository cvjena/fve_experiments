#!/usr/bin/env python

from setuptools import setup, find_packages

import fve_layer
install_requires = [line.strip() for line in open("requirements.txt").readlines()]

setup(
	name='fve_layer',
	python_requires=">3.7",
	version=fve_layer.__version__,
	description='Deep Fisher Vector Layer Implementation',
	author='Dimitri Korsch',
	author_email='korschdima@gmail.com',
	license='MIT License',
	packages=find_packages(),
	zip_safe=False,
	setup_requires=[],
	install_requires=install_requires,
	package_data={'': ['requirements.txt']},
	data_files=[('.',['requirements.txt'])],
	exclude_package_data={
		"": ["example*", "generated_data", "notebooks", "tests", "htmlcov"]
	},
	include_package_data=True,
)
