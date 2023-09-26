#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from cvargparse import Arg
from cvargparse import BaseParser
from scipy import stats

from chainer_addons.utils.sacred import MPIExperiment
from cvfinetune.utils.sacred_plotter import SacredPlotter


Setup = namedtuple("Setup", ["parts", "fve_type", "shuffle_parts", "aggregation"], defaults=[False, "concat"])


class ConfidenceInterval:

	def __init__(self, confidence: float):
		super().__init__()
		self.confidence = confidence

	def t(self, dof: int):
		return stats.t.ppf((1 + self.confidence) / 2, dof)

	def __call__(self, values):
		sem = stats.sem(values)
		return self.t(len(values)-1) * sem

	def __repr__(self):
		return f"<{type(self).__name__} with confidence of {self.confidence:.2%}>"



class Factory:

	input_sizes = {
		# model_type: [SMALL_SIZE, BIG_SIZE]
		"chainercv2.resnet18": [224, 448],
		"chainercv2.resnet50": [224, 448],
		"cvmodelz.InceptionV3": [299, 427],
	}

	def __init__(self, args):
		self._args = args


		self.plus_minus = dict(
			stddev=np.std,
			stderr=stats.sem,
			conf68=ConfidenceInterval(0.68),
			conf95=ConfidenceInterval(0.95),
		)[args.plus_minus]

		logging.info(f"Using {self.plus_minus} as \u00B1 value")

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
			"config.learning_rate": 4e-3,
			"config.n_components": self._args.n_comps,
			"config.comp_size": self._args.comp_size,
		}

	def setup2label(self, setup: Setup, values):
		shuffled = "|shuffled" if setup.shuffle_parts else ""
		setup_str = f"{setup.parts} ({setup.fve_type}{shuffled})\nAgg: {setup.aggregation}"
		plus_minus = self.plus_minus(values)
		accu_str = f"{np.mean(values):.1%}(\u00B1{plus_minus:.1%})"
		return "\n".join([
			setup_str,
			accu_str,
			f"[{len(values)} runs]"])

	def __call__(self, setup: Setup):
		query = self.default_query

		query["config.parts"] = setup.parts
		query["config.feature_aggregation"] = setup.aggregation

		if setup.fve_type != "ALL":
			query["config.shuffle_parts"] = setup.shuffle_parts
			query["config.fve_type"] = setup.fve_type

			if setup.fve_type in ["em", "grad"]:
				query["config.aux_lambda"] = 0.0

		return query

def main(args):
	creds = MPIExperiment.get_creds()
	plotter = SacredPlotter(creds)

	factory = Factory(args)

	fig = plt.figure()
	fig.suptitle("\n".join([
		args.dataset,
		f"{args.model_type.split('.')[-1]}@{factory.input_size}px",
		f"\u00B1: {args.plus_minus}",
	]))
	plotter.plot(
		metrics=args.metrics,
		setups=[
			# Setup("GT2", "ALL"),
			Setup("GT2", "no"),
			Setup("GT2", "no", True),
			Setup("GT2", "no", aggregation="mean"),
			Setup("GT2", "no", True, aggregation="mean"),
			Setup("GT2", "em"),
			Setup("GT2", "em", True),
			# Setup("L1_pred", "ALL"),
			# Setup("L1_pred", "no"),
			# Setup("L1_pred", "em"),
			# Setup("L1_pred", "grad"),
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
		choices=["accu", "g_accu", "p_accu", "dist"], nargs="+"),

	Arg("--outliers", action="store_true"),
	Arg("--include_running", action="store_true"),

	Arg("--dataset", "-ds", default="CUB200",
		choices=["CUB200", "NAB", "BIRDSNAP", "EU_MOTHS", "DOGS", "CARS"]),

	Arg("--model_type", "-mt", default="cvmodelz.InceptionV3",
		choices=["cvmodelz.InceptionV3", "chainercv2.resnet50", "chainercv2.resnet18"]),

	Arg("--big", "-big", action="store_true"),
	Arg("--n_comps", "-nc", default=1, type=int),
	Arg("--comp_size", "-fve_size", default=0, type=int),
	Arg("--optimizer", "-opt", default="adam", choices=["adam", "rmsprop", "sgd"]),
	Arg("--plus_minus", "-pm", default="conf95", choices=["stderr", "stddev", "conf68", "conf95"]),

])

main(parser.parse_args())
