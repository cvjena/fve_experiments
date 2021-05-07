import copy
import chainer
import numpy as np

from chainer import functions as F
from chainer.backends import cuda
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

from gen_data import data as data_module
from gen_data import utils
from gen_data.classifier.fve import FVEClassifier
from gen_data.classifier.training import train


def _soft_assignment(data, mean, var, w):
	n_comps = mean.shape[-1]
	_gmm = GMM(n_components=n_comps, covariance_type="diag", warm_start=True)

	_gmm.covariances_ = var.T
	_gmm.precisions_cholesky_ = 1 / np.sqrt(var.T)
	_gmm.means_ = mean.T
	_gmm.weights_ = w

	return _gmm.predict_proba(data)

def dist(data, mean, var, w=None):
	# print("Input shapes:", data.shape, mean.shape, var.shape)
	_data = data[..., None]
	n_comps = mean.shape[-1]
	_mean = mean[None]
	_var = var[None]

	if w is None:
		w = np.ones(n_comps, dtype=_mean.dtype) / n_comps

	dist = np.sqrt( ((_data - _mean)**2 / _var).sum(axis=1) )
	gamma = _soft_assignment(data, mean, var, w)

	return (dist * gamma).sum(axis=-1).mean()

def analyze_data(data: data_module.Data):
	# Mahalanobis Distances
	_data = utils.get_array(data.X)

	# ... to mean and variance of the Data
	_mean = _data.mean(axis=0)[:, None]
	_var = _data.var(axis=0)[:, None]
	dist0 = dist(_data, _mean, _var)

	# ... to means and stds used for data generation
	_mean = data._means.T
	_var = data._std[:, None].repeat(data.n_classes, axis=-1)**2
	dist1 = dist(_data, _mean, _var)

	# ... to means and stds estimated by an offline GMM
	gmm = GMM(n_components=data.n_classes, covariance_type="diag")
	gmm.fit(_data)

	_means = gmm.means_.T
	_var = gmm.covariances_.T
	_w = gmm.weights_

	dist2 = dist(_data, _means, _var, _w)

	return float(dist0), float(dist1), float(dist2)


def analyze_data_change(args, data: data_module.Data, clf: FVEClassifier, *,
						triggers: dict,
						eval_data: data_module.Data,
						plot_decisions: bool = False,
						no_plot: bool = False,
						**kwargs,):
	if data.X.shape[1] != 2: return

	new_data = copy.deepcopy(data)

	fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16,9))

	device = cuda.get_device_from_id(args.device)
	device.use()

	if args.device >= 0:
		clf.to_device(device)
		data.X.to_device(device)
		data.y = cuda.to_gpu(data.y)

	train(data, clf,
		  eval_data=eval_data,
		  batch_size=args.batch_size,
		  learning_rate=args.learning_rate,
		  device=args.device,
		  triggers=triggers,
		)

	clf.to_cpu()
	data.X.to_cpu()
	data.y = cuda.to_cpu(data.y)

	ax0.set_title("Initial data")
	utils.plotting._plot_params(data, clf,
				 # eval_data=eval_data,
				 fig_axs=(fig, ax0),
				 **kwargs)


	for epoch in range(args.epochs):
		new_data.X.cleargrad()

		with chainer.using_config("train", False):
			loss = clf(new_data.X, new_data.y)

		loss.backward()
		assert new_data.X.grad is not None, "backward did not work!"

		new_data.X.array -= F.normalize(new_data.X.grad).array

	ax1.set_title("Updated data")
	utils.plotting._plot_params(new_data, clf,
				 # eval_data=eval_data,
				 fig_axs=(fig, ax1),
				 **kwargs)



	if plot_decisions:
		X, y = data.X.array, data.y
		with chainer.using_config("train", False):
			utils.plotting._plot_decisions(X, y, clf=clf, alpha=0.3, ax=ax0)
			utils.plotting._plot_decisions(X, y, clf=clf, alpha=0.3, ax=ax1)




