#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import numpy as np
import re
import simplejson as json

from collections import defaultdict
from cvargparse import Arg
from cvargparse import BaseParser
from matplotlib import pyplot as plt
from munch import munchify
from pathlib import Path
from tabulate import tabulate
from typing import List

def group_outputs(outputs: List[Path], name_regex=re.compile(r"(\d+)_(\d+)_(\d+).json")):

	result = defaultdict(lambda: defaultdict(list))
	comps = set()
	dims = set()
	for output in outputs:
		match = name_regex.match(output.name)
		assert match is not None
		idx, n_comp, n_dim = match.groups()
		n_comp, n_dim = int(n_comp), int(n_dim)
		comps.add(n_comp)
		dims.add(n_dim)
		result[n_comp][n_dim].append(output)

	return dict(result), sorted(comps), sorted(dims)

def read_data(paths: List[Path]):
	accus = dict(baseline=[], em=[], grad=[])
	dists = dict(baseline=[], em=[], grad=[])
	for path in paths:
		with open(path) as f:
			cont = munchify(json.load(f))

		accus["baseline"].append(cont.svm_baseline.val_accu)
		accus["em"].append(cont.fveEM.val_accu)
		accus["grad"].append(cont.fveGrad.val_accu)

		dists["baseline"].append(cont.eval_data[-1])
		dists["em"].append(cont.fveEM.val_dist)
		dists["grad"].append(cont.fveGrad.val_dist)

	return accus, dists

def main(args):
	folder = Path(args.folder)

	outputs = folder.glob("*.json")

	grouped, comps, dims = group_outputs(outputs)

	fig, axs = plt.subplots(ncols=2)

	for n_comp, ax in zip(comps, axs):
		_rows = dict(baseline=[], em=[], grad=[])
		for n_dim in dims:
			accus, dists = read_data(grouped[n_comp][n_dim])

			for name in ["baseline", "em", "grad"]:
				accu, dist = accus[name], dists[name]
				_rows[name].append(accu)

		rows = []
		for name in ["baseline", "em", "grad"]:
			values = np.array(_rows[name])
			rows.append([name] + [f"{mean:.2%} (+/- {std:.2%})" for mean, std in zip(values.mean(axis=-1), values.std(axis=-1))])

		ax.set_title(f"{n_comp} components")
		print(f"===== # components: {n_comp} =====")
		print(tabulate(rows, headers=dims, tablefmt="fancy_grid"))

	plt.show()
	plt.close()

parser = BaseParser()

parser.add_args([
	Arg("folder"),
])

main(parser.parse_args())
