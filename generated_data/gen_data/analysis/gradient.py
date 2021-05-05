import chainer
import numpy as np

from chainer import functions as F
from chainer.backends import cuda
from matplotlib import pyplot as plt

from gen_data import data as data_module
from gen_data import utils
from gen_data.classifier.base import Classifier
from gen_data.classifier.training import train


def analyze_gradient(args, data: data_module.Data, clf: Classifier,
					 triggers: dict,
					 eval_data: data_module.Data = None,
					 plot_params: bool = True,
					 plot_decisions: bool = False,
					 title: str = None,
					 **kwargs,
					):
	print(clf)

	train(data, clf,
		  batch_size=args.batch_size,
		  learning_rate=args.learning_rate,
		  triggers=triggers,
		  eval_data=eval_data)

	if data.X.shape[1] != 2: return

	fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16,9))

	feat = clf.encode(data.X)
	feat.grad = clf.xp.ones_like(feat.array)
	feat.backward()

	ax0.set_title("Gradient of the encoding wrt input")
	utils.plotting._plot_params(data, clf,
				 eval_data=eval_data,
				 plot_grad=True,
				 fig_axs=(fig, ax0),
				 **kwargs)



	data.X.grad = None
	feat.grad = None
	pred = clf.classify(feat)
	loss = F.softmax_cross_entropy(pred, data.y)
	loss.backward()

	ax1.set_title("Gradient of loss wrt input")
	utils.plotting._plot_params(data, clf,
				 eval_data=eval_data,
				 plot_grad=True,
				 fig_axs=(fig, ax1),
				 **kwargs)

	if plot_decisions:
		X, y = data.X.array, data.y
		with chainer.using_config("train", False):
			utils.plotting._plot_decisions(X, y, clf=clf, alpha=0.3, ax=ax1)

