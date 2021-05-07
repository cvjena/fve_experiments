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

	fig, axs = plt.subplots(nrows=len(comps), ncols=2)

	for n_comp, _axs in zip(comps, axs.T):
		_rows = [dict(baseline=[], em=[], grad=[]), dict(baseline=[], em=[], grad=[])]
		for n_dim in dims:
			accus, dists = read_data(grouped[n_comp][n_dim])

			for name in ["baseline", "em", "grad"]:
				accu, dist = accus[name], dists[name]
				_rows[0][name].append(accu)
				_rows[1][name].append(dist)

		rows = [[], []]

		accu_labels = dict(baseline="Linear SVM", em="CNN + FVELayer (em)", grad="CNN + FVELayer (grad)")
		dist_labels = dict(baseline="GMM", em="CNN + FVELayer (em)", grad="CNN + FVELayer (grad)")
		for name in ["baseline", "em", "grad"]:
			_аccus = np.array(_rows[0][name])
			_dists = np.array(_rows[1][name])

			mean_accu, std_accu = _аccus.mean(axis=-1), _аccus.std(axis=-1)
			min_accu, max_accu = _аccus.min(axis=-1), _аccus.max(axis=-1)
			mean_dist, std_dist = _dists.mean(axis=-1), _dists.std(axis=-1)
			min_dist, max_dist = _dists.min(axis=-1), _dists.max(axis=-1)

			rows[0].append([name] + [f"{mean:.2%} (+/- {std:.2%})" for mean, std in zip(mean_accu, std_accu)])
			rows[1].append([name] + [f"{mean:.2f} (+/- {std:.2f})" for mean, std in zip(mean_dist, std_dist)])

			# ax.boxplot(_аccus.T * 100)
			_axs[0].plot(mean_accu*100, label=accu_labels[name])
			_axs[0].fill_between(range(len(mean_accu)), min_accu*100, max_accu*100, alpha=0.3)
			_axs[0].legend()
			_axs[0].set_title(f"{n_comp} component{'s' if n_comp > 1 else ''}\nAccuracy (in %)")

			_axs[1].plot(mean_dist, label=dist_labels[name])
			_axs[1].fill_between(range(len(mean_dist)), min_dist, max_dist, alpha=0.3)
			_axs[1].set_yscale("log")
			_axs[1].set_title(f"{n_comp} component{'s' if n_comp > 1 else ''}\nMahalanobis distance")
			_axs[1].legend()

			[ax.set_xticklabels([""] + dims) for ax in _axs]
			[ax.set_xlabel("Feature dimensionality") for ax in _axs]

		print(f"===== # components: {n_comp} =====")

		print(f"=== Accuracy ===")
		print(tabulate(rows[0], headers=dims, tablefmt="fancy_grid"))

		print(f"=== Mahalanobis distance ===")
		print(tabulate(rows[1], headers=dims, tablefmt="fancy_grid"))

	plt.tight_layout()
	plt.show()
	plt.close()

parser = BaseParser()

parser.add_args([
	Arg("folder"),
])

main(parser.parse_args())
