#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from cvargparse import Arg
from cvargparse import BaseParser

from chainer_addons.utils.sacred import MPIExperiment
from cvfinetune.utils.sacred_plotter import SacredPlotter


Setup = namedtuple("Setup", ["parts", "fve_type"])

class Factory:

	input_sizes = {
		# model_type: [SMALL_SIZE, BIG_SIZE]
		"chainercv2.resnet50": [224, 448],
		"cvmodelz.InceptionV3": [299, 427],
	}

	def __init__(self, args):
		self._args = args

	@property
	def input_size(self):
		sizes = self.input_sizes[self._args.model_type]
		return sizes[1 if self._args.big else 0]

	@property
	def default_query(self):

		return {
			"experiment.name": "FVE Layer",
			"config.dataset": self._args.dataset,
			"config.model_type": self._args.model_type,
			"config.input_size": self.input_size,

			"config.optimizer": "adam",
			"config.feature_aggregation": "concat",
			"config.n_components": self._args.n_comps,
			"config.comp_size": self._args.comp_size,
		}

	def setup2label(self, setup: Setup, values):
		setup_str = f"{setup.parts} ({setup.fve_type})"
		accu_str = f"{np.mean(values):.2%}(\u00B1{np.std(values):.2%})"
		return "\n".join([setup_str, accu_str, f"[{len(values)} runs]"])

	def __call__(self, setup: Setup):
		query = self.default_query

		query["config.parts"] = setup.parts
		if setup.fve_type != "ALL":
			query["config.fve_type"] = setup.fve_type

		return query

def main(args):
	creds = MPIExperiment.get_creds()
	plotter = SacredPlotter(creds)

	factory = Factory(args)

	fig = plt.figure()
	fig.suptitle(f"{args.dataset}\n{args.model_type.split('.')[-1]} {factory.input_size}px")
	plotter.plot(
		metrics=args.metrics,
		setups=[
			Setup("L1_pred", "ALL"),
			Setup("L1_pred", "no"),
			Setup("L1_pred", "em"),
			Setup("L1_pred", "grad"),
		],

		query_factory=factory,
		setup_to_label=factory.setup2label,

		include_running=args.include_running,
		showfliers=args.outliers,
	)
	plt.show()
	plt.close()


parser = BaseParser()

parser.add_args([
	Arg("--metrics", "-m", default=["g_accu", "p_accu", "accu"],
		choices=["accu", "g_accu", "p_accu"], nargs="+"),

	Arg("--outliers", action="store_true"),
	Arg("--include_running", action="store_true"),

	Arg("--dataset", "-ds", default="CUB200",
		choices=["CUB200", "NAB", "BIRDSNAP", "EU_MOTHS", "DOGS", "CARS"]),

	Arg("--model_type", "-mt", default="cvmodelz.InceptionV3",
		choices=["cvmodelz.InceptionV3", "chainercv2.resnet50", ]),

	Arg("--big", "-big", action="store_true"),
	Arg("--n_comps", "-nc", default=1, type=int),
	Arg("--comp_size", "-fve_size", default=-1, type=int),
	Arg("--optimizer", "-opt", default="adam", choices=["adam", "rmsprop", "sgd"]),

])

main(parser.parse_args())
