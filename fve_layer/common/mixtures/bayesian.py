import numpy as np

from chainer.backends import cuda
from sklearn.mixture import BayesianGaussianMixture

from fve_layer.common.mixtures.base import GPUMixin

class BayesianGMM(GPUMixin, BayesianGaussianMixture):

	def _m_step(self, X, log_resp, xp=np):

		nk, means, covariances = \
			self._gaussian_params(X, log_resp, xp=xp)

		# estimate weights
		self.weight_concentration_ = self.weight_concentration_prior_ + nk
		self.weights_ = self.weight_concentration_ / self.weight_concentration_.sum()

		# estimate means
		self.mean_precision_ = self.mean_precision_prior_ + nk
		self.means_ = ((self.mean_precision_prior_ * self.mean_prior_ +
						nk[:, None] * means) /
					   self.mean_precision_[:, None])

		# estimate precisions
		_, n_features = means.shape

		# Warning : in some Bishop book, there is a typo on the formula 10.63
		# `degrees_of_freedom_k = degrees_of_freedom_0 + Nk`
		# is the correct formula
		self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

		diff = means - self.mean_prior_
		self.covariances_ = (
			self.covariance_prior_ + nk[:, None] * (
				covariances + (self.mean_precision_prior_ /
					  self.mean_precision_)[:, None] * xp.square(diff)))

		# Contrary to the original bishop book, we normalize the covariances
		self.covariances_ /= self.degrees_of_freedom_[:, None]
		self.precisions_cholesky_ = 1. / xp.sqrt(self.covariances_)

	def _check_parameters(self, X):
		super(BayesianGMM, self)._check_parameters(cuda.to_cpu(X))
		xp = self.xp_from_array(X)

		self.mean_prior_ = xp.array(self.mean_prior_)
		self.covariance_prior_ = xp.array(self.covariance_prior_)

	def _estimate_weights(self, nk):
		super(BayesianGMM, self)._estimate_weights(nk)
		self.weights_ = self.weight_concentration_ / self.weight_concentration_.sum()

	# def fit(self, X, y=None):
	# 	X, xp = self._tranform_X(X)
	# 	return super(BayesianGaussianMixture, self).fit(X, y)

