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


def set_seed(seed: int = None):
	np.random.seed(seed)
	cupy.random.seed(seed)
	random.seed(seed)

def main(args):
	set_seed(args.seed)

	data = data_module.Data.new(args)
	eval_data = data_module.Data.new(args, evaluation=True)
	if eval_data is not None:
		eval_data.X = eval_data.X.array

	result = dict()

	data_records = result["data_records"] = analysis.analyze_data(data)
	print("Mahalanobis distance of the data: {: 12.4f} | {: 12.4f} | {: 12.4f}\t".format(*data_records))

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

		set_seed(args.seed)

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

		clfs, records = analysis.analyze_classifier(plot_params=True, **kwargs)
		result[name] = records

		# if 1:
		# else:
		# 	analysis.analyze_gradient(plot_norm_grad=True, **kwargs)


	print(pyaml.dump(result, indent=2, sort_keys=False))
	print(json.dumps(result, indent=2))
	if args.output:
		with open(args.output, "w") as out:
			json.dump(result, out, indent=2)

	plt.show()
	plt.close()


main(parser.parse_args())