#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import yaml
import simplejson as json
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

def _check_filter(opts, filter_keys: dict):

	for key, value in filter_keys.items():

		if key in opts:
			if isinstance(value, str) and value.startswith("!="):
				if value[2:] == str(opts[key]):
					return False

			elif opts[key] != value:
				return False

	return True


def remove_outliers(values):
	mean_value = np.mean(values)

	# print(mean_value)
	# print(*zip(values, (values - mean_value)**2), sep="\n")

	values = [v for v in values if (v - mean_value)**2 < 0.1]

	return values

def main(args):
	Key = namedtuple("Key", args.group_by)

	filter_keys = {}
	if args.filter is not None:

		if args.filter.endswith(".yml"):
			filter_keys = yaml.safe_load(open(args.filter))

		elif args.filter.endswith(".json"):
			filter_keys = json.load(open(args.filter))

		else:
			raise ValueError(f"Unknown filter file type: {args.filter}")
	print(filter_keys)

	for folder in args.folders:
		folder = Path(folder)

		res = Result()
		for eval_file in folder.glob("**/log"):
			args_file = eval_file.parent / "meta/args.yml"
			assert args_file.exists()

			with open(eval_file) as f0, open(args_file) as f1:

				evals, opts = json.load(f0)[-1], yaml.safe_load(f1)
				# evals, opts = yaml.safe_load(f0), yaml.safe_load(f1)

			if not _check_filter(opts, filter_keys):
				continue

			metrics = {m: evals.get(m) for m in args.metrics}

			key = Key(*[_unpack(opts.get(m)) for m in args.group_by])
			res.insert(key, metrics)

		rows = []
		keys = sorted(res.keys())
		for key in keys:
			if key.fve_type not in ["em", "grad"]:
				continue

			row = [" ".join([str(k) for k in key])]
			for metric_key, values in res[key].items():
				values = [v for v in values if v is not None]
				if args.no_outliers:
					values = remove_outliers(values)

				if "accu" in metric_key:
					row.append(f"{np.mean(values):.2%} \u00b1 {np.std(values):.2%}")
				else:
					row.append(f"{np.mean(values):.3f} \u00b1 {np.std(values):.3f}")

				row[-1] += f" ({len(values)} runs)"

			if args.print_all:
				for metric_key, values in res[key].items():
					row.append(", ".join(f"{v:.3f}" for v in values))

			rows.append(row)

		headers = args.metrics
		if args.print_all:
			headers += [f"{m} (all)" for m in args.metrics]

		print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

parser = BaseParser([
	Arg("folders", nargs="+"),

	Arg("--metrics", "-m", nargs="+", default=["main/dist", "main/logL"]),
	Arg("--group_by", "-g", nargs="+", default=["dataset", "model_type", "fve_type", "input_size"]),

	Arg("--filter", "-f"),


	Arg("--no_outliers", action="store_true"),
	Arg("--print_all", "-all", action="store_true"),
])
main(parser.parse_args())
