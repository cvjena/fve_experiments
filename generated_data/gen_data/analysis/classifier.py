import chainer
import numpy as np

from chainer import functions as F
from chainer.backends import cuda
from matplotlib import pyplot as plt
from tabulate import tabulate

from gen_data import data as data_module
from gen_data import utils
from gen_data.classifier.fve import FVEClassifier
from gen_data.classifier.training import train

def _clf_array(clf: chainer.Chain, arr):
	return clf.xp.array(utils.get_array(arr))

def calc_accu(clf: FVEClassifier, data: data_module.Data):
	decisions = clf.decision_function(_clf_array(clf, data.X))
	decisions = cuda.to_cpu(utils.get_array(decisions))
	return (decisions.argmax(axis=-1) == cuda.to_cpu(data.y)).mean()

def calc_dist(clf: FVEClassifier, data: data_module.Data):
	X = _clf_array(clf, data.X)
	x = clf.xp.expand_dims(X, axis=1)
	return utils.get_array(clf.distance(x))

def evaluations(clf: FVEClassifier, data: data_module.Data, eval_data: data_module.Data):

	accu = calc_accu(clf, data)
	val_accu = calc_accu(clf, eval_data)

	dist = calc_dist(clf, data)
	val_dist = calc_dist(clf, eval_data)


	return dict(
		accu=float(accu), val_accu=float(val_accu),
		dist=float(dist), val_dist=float(val_dist),
		comp_weights=utils.get_array(clf.fve_layer.w).tolist(),
	)


def analyze_classifier(args, data: data_module.Data, clf: FVEClassifier, *,
					   triggers: dict,
					   eval_data: data_module.Data = None,
					   no_plot: bool = True,
					   print_params: bool = False,
					   plot_params: bool = True,
					   plot_decisions: bool = False,
					   title: str = None):
	# print(clf)
	if data.X.shape[1] == 2:
		clf_dump = clf.copy(mode="copy")


	device = cuda.get_device_from_id(args.device)
	device.use()

	if args.device >= 0:
		clf.to_device(device)
		data.X.to_device(device)
		data.y = cuda.to_gpu(data.y)

	train(data, clf,
		  batch_size=args.batch_size,
		  learning_rate=args.learning_rate,
		  device=args.device,
		  triggers=triggers,
		  eval_data=eval_data)

	with chainer.using_config("train", False), chainer.no_backprop_mode():
		result = evaluations(clf, data, eval_data)

	clf.to_cpu()
	data.X.to_cpu()
	data.y = cuda.to_cpu(data.y)


	if data.X.shape[1] == 2 and not no_plot:

		if plot_params:

			fig, (ax0, ax1) = utils.plotting._plot_params(data, clf,
										   eval_data=eval_data,
										   clf_dump=clf_dump,
										   title=title)

		else:
			ax0 = None
			fig, ax1 = plt.subplots(figsize=(12,12))
			data.plot(ax1)
			if eval_data is not None:
				eval_data.plot(ax1, marker="x", alpha=0.5)

		if plot_decisions:
			X, y = data.X.array, data.y
			with chainer.using_config("train", False):
				#utils.plotting._plot_decisions(X, y, clf=clf_dump, alpha=0.3, ax=ax0)
				utils.plotting._plot_decisions(X, y, clf=clf, alpha=0.3, ax=ax1)

		if print_params:
			fmt = "({0: 10.3f}, {1: 10.3f})"

			if args.n_components == 1:
				_x = utils.get_array(data.X)
				print("=== Data stats ===")
				print(tabulate([(fmt.format(*_x.mean(axis=0)), fmt.format(*_x.var(axis=0)))],
							   headers=["\u03BC", "\u03C3"],
							   tablefmt="fancy_grid",
							  )
					 )
				print()

			for i, c in enumerate([clf_dump, clf]):
				rows = []

				for _mu, _sig in zip(utils.get_array(c.fve_layer.mu).T, utils.get_array(c.fve_layer.sig).T):
					rows.append([fmt.format(*_mu), fmt.format(*_sig)])

				print("=== Estimated GMM Parameters ===" if i == 1 else "=== Initial GMM Parameters ===")
				print(tabulate(rows,
							   headers=["\u03BC", "\u03C3"],
							   tablefmt="fancy_grid",
							   showindex=True,
							  )
					 )
				print()


	return clf, result

