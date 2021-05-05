import numpy as np

from sklearn.mixture import GaussianMixture as GMM

from gen_data import data as data_module
from gen_data import utils


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
