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

def calc_entropy(arr: np.ndarray):
	return -np.sum(arr * np.log(arr)).round(5)

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
	entropies = dict(em=[], grad=[])
	for path in paths:
		with open(path) as f:
			cont = munchify(json.load(f))

		accus["baseline"].append(cont.svm_baseline.val_accu)
		accus["em"].append(cont.fveEM.val_accu)
		accus["grad"].append(cont.fveGrad.val_accu)

		dists["baseline"].append(cont.eval_data[-1])
		dists["em"].append(cont.fveEM.val_dist)
		dists["grad"].append(cont.fveGrad.val_dist)

		entropies["em"].append(calc_entropy(cont.fveEM.comp_weights))
		entropies["grad"].append(calc_entropy(cont.fveGrad.comp_weights))

	return accus, dists, entropies

def main(args):
	folder = Path(args.folder)

	outputs = folder.glob("*.json")

	grouped, comps, dims = group_outputs(outputs)

	fig, axs = plt.subplots(nrows=len(comps), ncols=2)
	ent_fig, ent_axs = plt.subplots(nrows=len(comps))

	for i, n_comp in enumerate(comps):
		_rows = dict(baseline=[], em=[], grad=[]), dict(baseline=[], em=[], grad=[])
		ent_rows = dict(em=[], grad=[])
		for n_dim in dims:
			accus, dists, entropies = read_data(grouped[n_comp][n_dim])

			for name in ["baseline", "em", "grad"]:
				_rows[0][name].append(accus[name])
				_rows[1][name].append(dists[name])

				if name != "baseline":
					ent_rows[name].append(entropies[name])

		rows = [[], [], []]

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
			axs[i][0].plot(mean_accu*100, label=accu_labels[name])
			axs[i][0].fill_between(range(len(mean_accu)), min_accu*100, max_accu*100, alpha=0.3)
			axs[i][0].legend()
			axs[i][0].set_title(f"{n_comp} component{'s' if n_comp > 1 else ''}\nAccuracy (in %)")

			axs[i][1].plot(mean_dist, label=dist_labels[name])
			axs[i][1].fill_between(range(len(mean_dist)), min_dist, max_dist, alpha=0.3)
			axs[i][1].set_yscale("log")
			axs[i][1].set_title(f"{n_comp} component{'s' if n_comp > 1 else ''}\nMahalanobis distance")
			axs[i][1].legend()

			[ax.set_xticklabels([""] + dims) for ax in axs[i]]
			[ax.set_xlabel("Feature dimensionality") for ax in axs[i]]

			if name == "baseline":
				continue

			_ents = np.array(ent_rows[name])
			mean_ent, std_ent = _ents.mean(axis=-1), _ents.std(axis=-1)
			min_ent, max_ent = _ents.min(axis=-1), _ents.max(axis=-1)
			rows[2].append([name] + [f"{mean:.2f} (+/- {std:.2f})" for mean, std in zip(mean_ent,  std_ent)])

			ent_axs[i].plot(mean_ent, label=accu_labels[name])
			ent_axs[i].fill_between(range(len(mean_ent)), min_ent, max_ent, alpha=0.3)
			ent_axs[i].set_title(f"{n_comp} component{'s' if n_comp > 1 else ''}\nMahalanobis distance")
			ent_axs[i].legend()
			ent_axs[i].set_xlabel("Feature dimensionality")
			ent_axs[i].set_xticklabels([""] + dims)


		print(f"===== # components: {n_comp} =====")

		print(f"=== Accuracy ===")
		print(tabulate(rows[0], headers=dims, tablefmt="fancy_grid"))

		print(f"=== Mahalanobis distance ===")
		print(tabulate(rows[1], headers=dims, tablefmt="fancy_grid"))

		print(f"=== Entropy ===")
		print(tabulate(rows[2], headers=dims, tablefmt="fancy_grid"))

	plt.tight_layout()
	plt.show()
	plt.close()

parser = BaseParser()

parser.add_args([
	Arg("folder"),
])

main(parser.parse_args())
