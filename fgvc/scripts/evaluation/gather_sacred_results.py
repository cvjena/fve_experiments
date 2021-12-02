#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import numpy as np

from chainer_addons.utils.sacred import MPIExperiment
from cvargparse import Arg
from cvargparse import BaseParser
from cvfinetune.utils.sacred_plotter import SacredPlotter
from collections import namedtuple
from matplotlib import pyplot as plt

Setup1 = namedtuple("Setup", ["parts", "fve_type", "comp_size", "feature_agg"])
Setup2 = namedtuple("Setup", ["parts", "fve_type", "shuffle_parts"])

class Factory(object):

	def __init__(self, args):

		self.dataset = args.dataset
		self.model_type = args.model_type
		self.feature_agg = args.feature_agg

	@property
	def default_query(self):
		return {
			"experiment.name": "FVE Layer",
			"config.dataset": self.dataset,
			"config.model_type": self.model_type,
			"config.feature_aggregation": self.feature_agg,

			# "config.aux_lambda": 0.5,
			"config.optimizer": "adam",
			"config.from_scratch": False,
		}

	def __call__(self, setup):

		if isinstance(setup, Setup1):
			query = {
				"config.parts": setup.parts,
				"config.fve_type": setup.fve_type,
				"config.comp_size": setup.comp_size,
			}

			if setup.feature_agg is not None:
				query["config.feature_aggregation"] = setup.feature_agg

		elif isinstance(setup, Setup2):
			query = {
				"config.parts": setup.parts,
				"config.fve_type": setup.fve_type,
				"config.shuffle_parts": setup.shuffle_parts,
				"config.optimizer": "rmsprop" if setup.parts == "GT" else "adam"
			}

		return dict(self.default_query, **query)

	def setup_to_label(self, setup, values):

		if isinstance(setup, Setup1):
			setup_str = f"{setup.parts} ({setup.fve_type})"

			if setup.feature_agg is not None:
				setup_str += f"\nfeature_agg={setup.feature_agg}"

			msg = "\t".join([
				f"{setup_str}",
				f"({len(values)} runs)",
				f"{np.mean(values):.3%} (\u00B1 {np.std(values):.3%})",
			])

			print(msg)

			return "\n".join([
				f"{setup_str}",
				f"{np.mean(values):.2%} (\u00B1  {np.std(values):.2%})",
				f"({len(values)} runs)",
			])

		elif isinstance(setup, Setup2):
			setup_str = f"{15 if setup.parts == 'GT' else 4} GT Parts"
			msg = "\t".join([
				f"{setup_str}",
				f"shuffled={setup.shuffle_parts}",
				f"({len(values)} runs)",
				f"{np.mean(values):.3%} (\u00B1 {np.std(values):.3%})",
			])
			print(msg)
			return "\n".join([
				f"{setup_str}",
				f"shuffled={setup.shuffle_parts}",
				f"{np.mean(values):.2%} (\u00B1 {np.std(values):.2%})",
				f"({len(values)} runs)",
			])

all_setups = [

	[
		# CS-Parts with GAP, EM-FVE, and Grad-FVE comparison
		Setup1("L1_pred", "no", -1, None),
		Setup1("L1_pred", "em", -1, None),
		Setup1("L1_pred", "grad", -1, None),
		# Setup1("L1_pred", "no", -1, "mean"),
		# Setup1("L1_pred", "em", -1, "mean"),
		# Setup1("L1_pred", "grad", -1, "mean"),
	],

	[
		# shuffled on GT parts
		Setup2("GT2", "no", True),
		Setup2("GT2", "no", False),

		# Setup2("GT", "no", True),
		# Setup2("GT", "no", False),
	],



]


def main(args):
	creds = MPIExperiment.get_creds()

	plotter = SacredPlotter(creds)


	fig = plt.figure()

	fig.suptitle(f"{args.dataset} - {args.model_type.split('.')[-1]}")
	factory = Factory(args)

	plotter.plot(
		metrics=args.metrics,

		setups=all_setups[1],
		query_factory=factory,
		setup_to_label=factory.setup_to_label,

		include_running=args.include_running,

		# plot_kwargs
		showfliers=args.outliers
	)

	plt.show()
	plt.close()


parser = BaseParser([

	Arg("--metrics", "-m", nargs="+", default=["accu"],
		choices=["accu", "p_accu", "g_accu"]),

	Arg("--dataset", "-ds", default="CUB200",
		choices=["CUB200", "EU_MOTHS"]),

	Arg("--feature_agg", default="concat",
		choices=["concat", "mean"]),

	Arg("--model_type", "-mt", default="cvmodelz.InceptionV3"),

	Arg("--outliers", action="store_true"),

	Arg("--include_running", action="store_true"),
])
main(parser.parse_args())
