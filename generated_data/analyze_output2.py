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

def group_outputs(outputs: List[Path], name_regex=re.compile(r"\w*(\d+)_\w*(\d+)_\w*(\d+).json")):

	result = defaultdict(lambda: defaultdict(list))
	comps = set()
	dims = set()
	for output in outputs:
		match = name_regex.match(output.name)
		assert match is not None
		n_comp, n_dim, idx = match.groups()
		n_comp, n_dim = int(n_comp), int(n_dim)
		comps.add(n_comp)
		dims.add(n_dim)
		result[n_comp][n_dim].append(output)

	return dict(result), sorted(comps), sorted(dims)



def read_data(paths: List[Path]):
	change = dict(em=[], grad=[])
	for path in paths:
		with open(path) as f:
			cont = munchify(json.load(f))

		change["em"].append((cont.fveEM.data_change_mean, cont.fveEM.data_change_std))
		change["grad"].append((cont.fveGrad.data_change_mean, cont.fveGrad.data_change_std))

	return change

def main(args):
	folder = Path(args.folder)

	outputs = folder.glob("*.json")

	grouped, comps, dims = group_outputs(outputs)

	fig, axs = plt.subplots(ncols=len(comps), nrows=1)

	for i, n_comp in enumerate(comps):
		_rows = dict(em=[], grad=[])
		for n_dim in dims:
			change = read_data(grouped[n_comp][n_dim])

			for name in ["em", "grad"]:
				_rows[name].append(change[name])

		rows = []

		accu_labels = dict(baseline="Linear SVM", em="FVELayer (em)", grad="FVELayer (grad)")
		dist_labels = dict(baseline="GMM", em="FVELayer (em)", grad="FVELayer (grad)")
		for name in ["em", "grad"]:

			_change = np.array(_rows[name]).transpose(2,0,1)
			mean_change, std_change = _change.mean(axis=-1)
			min_change, max_change = _change[0].min(axis=-1), _change[0].max(axis=-1)
			rows.append([name] + [f"{mean:.2f} (+/- {std:.2f})" for mean, std in zip(mean_change,  std_change)])

			axs[i].plot(mean_change, label=accu_labels[name])
			axs[i].fill_between(range(len(mean_change)), min_change, max_change, alpha=0.3)
			axs[i].set_title(f"{n_comp} component{'s' if n_comp > 1 else ''}")
			axs[i].legend()
			# axs[i].set_yscale("log")
			# axs[i].hlines(y=1, xmin=0, xmax=len(mean_change)-1, linestyle="dashed")
			axs[i].set_xlabel("Feature dimensionality")
			axs[i].set_xticklabels([""] + dims)

		# print(f"===== # components: {n_comp} =====")

		# print(f"=== Accuracy ===")
		# print(tabulate(rows[0], headers=dims, tablefmt="fancy_grid"))

		# print(f"=== Mahalanobis distance ===")
		# print(tabulate(rows[1], headers=dims, tablefmt="fancy_grid"))

		# print(f"=== Entropy ===")
		# print(tabulate(rows[2], headers=dims, tablefmt="fancy_grid"))

	plt.tight_layout()
	plt.show()
	plt.close()

parser = BaseParser()

parser.add_args([
	Arg("folder"),
])

main(parser.parse_args())
