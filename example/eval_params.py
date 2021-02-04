#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import numpy as np
import simplejson as json
import yaml
import pyaml

from collections import Counter
from collections import defaultdict
from pathlib import Path
from tabulate import tabulate

from cvargparse import Arg
from cvargparse import BaseParser

def main(args):
	folder = Path(args.results_folder)
	results_by_key = defaultdict(list)

	for opts_file in folder.glob("**/args.yml"):
		with open(opts_file, "r") as f:
			opts = yaml.safe_load(f)
			group_key = "_".join(map(str, map(opts.get, args.group_keys)))
			results_by_key[group_key].append(opts_file.with_name("clf_final.npz"))

	logging.info("Grouping values by: {}".format(", ".join(args.group_keys)))

	final_result = {}

	rows = []
	setups = sorted(results_by_key.keys())
	for setup in setups:
		row = [setup]
		param_dict = defaultdict(list)
		for dump in results_by_key.get(setup):
			if not dump.exists():
				logging.debug(f"skipping {dump}, because it does not exist")
				continue

			cont = np.load(dump)

			for name in args.param_names:

				param = cont.get(name)
				if param is None:
					continue
				param_dict[name].append(param)

		for name  in args.param_names:
			if len(param_dict[name]) == 0:
				row.extend(["-", "-"])
				continue
			params = np.array(param_dict[name])
			axis = tuple(np.arange(1, params.ndim))
			for func in [np.min, np.max]:
				_ = func(params, axis=axis)
				if len(_) == 1:
					row.append(f"{float(_):.4f}")
				else:
					row.append(f"{_.mean():.4f} \u00B1 {_.std():.4f}")

		rows.append(row)


	headers = []
	for name in args.param_names:
		headers.extend([f"{name} (min)", f"{name} (max)"])
	print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

parser = BaseParser()

parser.add_args([
	Arg("results_folder"),

	Arg("--group_keys", nargs="+",
		default=["fve_type", "dataset", "model_type", "parts", "n_components"]),

	Arg("--param_names", nargs="+",
		default=["fve_layer/mu", "fve_layer/sig"]),

])
main(parser.parse_args())
