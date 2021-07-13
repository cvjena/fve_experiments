#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import yaml
import numpy as np

from cvargparse import Arg
from cvargparse import BaseParser
from pathlib import Path
from collections import defaultdict
from collections import namedtuple
from functools import partial
from tabulate import tabulate


class Result(defaultdict):

	def __init__(self, factory=None, *args, **kwargs):
		if factory is None:
			factory = partial(defaultdict, list)

		super().__init__(factory, *args, **kwargs)

	def insert(self, key, metrics):
		for metric_key, value in metrics.items():
			self[key][metric_key].append(value)

def _unpack(obj):
	if isinstance(obj, list):
		obj = tuple(obj)

	if obj == False:
		obj = "no"
	return obj

def main(args):
	Key = namedtuple("Key", args.group_by)

	for folder in args.folders:
		folder = Path(folder)

		res = Result()
		for eval_file in folder.glob("**/evaluation.yml"):
			args_file = eval_file.parent / "meta/args.yml"
			assert args_file.exists()

			with open(eval_file) as f0, open(args_file) as f1:

				evals, opts = yaml.safe_load(f0), yaml.safe_load(f1)

			metrics = {m: evals.get(m, 0) for m in args.metrics}

			key = Key(*[_unpack(opts.get(m)) for m in args.group_by])
			res.insert(key, metrics)

		rows = []
		keys = sorted(res.keys())
		for key in keys:
			if key.fve_type not in ["em", "grad"]:
				continue

			row = [" ".join([str(k) for k in key])]
			for metric_key, values in res[key].items():
				if "accu" in metric_key:
					row.append(f"{np.mean(values):.2%} \u00b1 {np.std(values):.2%}")
				else:
					row.append(f"{np.mean(values):.3f} \u00b1 {np.std(values):.3f}")

			if args.print_all:
				for metric_key, values in res[key].items():
					row.append(", ".join(f"{v:.1f}" for v in values))

			rows.append(row)

		factor = 2 if args.print_all else 1
		print(tabulate(rows, headers=args.metrics * factor, tablefmt="fancy_grid"))

parser = BaseParser([
	Arg("folders", nargs="+"),

	Arg("--metrics", nargs="+", default=["main/dist", "main/logL"]),
	Arg("--group_by", nargs="+", default=["dataset", "model_type", "fve_type", "input_size"]),

	Arg("--print_all", action="store_true"),
])
main(parser.parse_args())
