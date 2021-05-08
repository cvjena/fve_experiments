#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import cupy
import numpy as np
import pyaml
import random
import simplejson as json
import warnings

from chainer.training import Trainer
from matplotlib import pyplot as plt

from fve_layer.backends.chainer.links import FVELayer
from fve_layer.backends.chainer.links import FVELayer_noEM

from gen_data import analysis
from gen_data import classifier
from gen_data import data as data_module
from gen_data.utils import parser


def set_seed(args):
	if args.seed is None:
		args.seed = np.random.randint(2**31-1)

	seed = args.seed
	np.random.seed(seed)
	cupy.random.seed(seed)
	random.seed(seed)

def main(args):
	set_seed(args)
	print(f"====== using seed {args.seed} =====")

	data = data_module.Data.new(args)
	eval_data = data_module.Data.new(args, evaluation=True)
	if eval_data is not None:
		eval_data.X = eval_data.X.array

	result = dict()

	if "data" in args.analyze:
		train_data_records = result["train_data"] = analysis.analyze_data(data)
		eval_data_records = result["eval_data"] = analysis.analyze_data(eval_data)
		print("Mahalanobis distances:")
		print("|____ training data: {: 12.4f} | {: 12.4f} | {: 12.4f}".format(*train_data_records))
		print("|__ evaluation data: {: 12.4f} | {: 12.4f} | {: 12.4f}".format(*eval_data_records))

	if "baseline" in args.analyze:
		svm, svm_records = classifier.baseline(data, eval_data, no_plot=args.no_plot)
		result["svm_baseline"] = svm_records

	iters_per_epoch = int(args.n_samples*args.n_classes / args.batch_size)
	for name, fve_class, epochs in [("fveEM", FVELayer, None), ("fveGrad", FVELayer_noEM, None)]:
		epochs = epochs or args.epochs

		triggers = dict(
			stop=(epochs, "epoch"),
			log=(max(1, int(epochs / 5)), "epoch"),
			progress_bar=iters_per_epoch*1
		)

		set_seed(args)

		clf = classifier.FVEClassifier.new(args,
			fve_class=fve_class, init_mu=args.init_mu, init_sig=args.init_sig)

		kwargs = dict(
			args=args,
			data=data,
			eval_data=eval_data,
			clf=clf,
			triggers=triggers,
			plot_decisions=True,
			no_plot=args.no_plot,
			title=fve_class.__name__
		)

		if "classifier" in args.analyze:
			print(clf)
			clf, records = analysis.analyze_classifier(plot_params=True, **kwargs)
			if clf.embedding is not None and hasattr(clf.embedding, "W"):
				print(clf.embedding.W)
			result[name] = records

		if "gradient" in args.analyze:
			analysis.analyze_gradient(plot_norm_grad=True, **kwargs)

		if "data_change" in args.analyze:
			analysis.analyze_data_change(**kwargs)


	print(pyaml.dump(result, indent=2, sort_keys=False))
	if args.output:
		with open(args.output, "w") as out:
			json.dump(result, out, indent=2)
	# print(json.dumps(result, indent=2))

	plt.show()
	plt.close()


main(parser.parse_args())
