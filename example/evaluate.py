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

from cvargparse import Arg
from cvargparse import BaseParser


"""
Example:
	- python evaluate.py .results --eval_keys val/main/accu val/main/g_accu val/main/p_accu val/main/aux_p_accu
"""

def main(args):
	folder = Path(args.results_folder)
	results_by_key = defaultdict(list)

	for opts_file in folder.glob("**/args.yml"):
		with open(opts_file, "r") as f:
			opts = yaml.safe_load(f)
			group_key = "_".join(map(str, map(opts.get, args.group_keys)))
			results_by_key[group_key].append(opts_file.with_name("log"))

	counts = [(key, len(values)) for key, values in results_by_key.items()]
	logging.info("Grouping values by: {}".format(", ".join(args.group_keys)))
	logging.debug(Counter(dict(counts)))

	final_result = {}


	for setup, logs in results_by_key.items():
		eval_values = defaultdict(list)
		for log in logs:
			if not log.exists():
				logging.debug(f"skipping {log}, because it does not exist")
				continue
			with open(log, "r") as f:
				log_cont = json.load(f)
			last = log_cont[-1]
			if args.only_epoch >= 1 and last["epoch"] < args.only_epoch:
				logging.debug(f"skipping {log}, because it has not enough epochs ({last['epoch']})")
				continue
			for key in args.eval_keys:
				eval_values[key].append(last.get(key, 0))

		setup_results = final_result.get(setup, {})
		for key, values in eval_values.items():
			if not values:
				logging.debug(f"Missing values for {key}!")
				continue
			setup_results[key] = f"{np.mean(values):.2%} +/- {np.std(values):.2%} ({len(values)} runs)"
		final_result[setup] = setup_results

	print(pyaml.dump(final_result))


parser = BaseParser()

parser.add_args([
	Arg("results_folder"),

	Arg("--group_keys", nargs="+", default=["fve_type"]),
	Arg("--eval_keys", nargs="+", default=["val/main/accu"]),
	Arg("--only_epoch", type=int, default=60),
])
main(parser.parse_args())
