import numpy as np

from sklearn.mixture import GaussianMixture

from fve_layer.common.mixtures.base import GPUMixin

class GMM(GPUMixin, GaussianMixture):


	def _m_step(self, X, log_resp, xp=np):
		""" M-Step
			Copied from sklearn/mixture/gaussian_mixture.py
		"""

		self.weights_, self.means_, self.covariances_ = \
			self._gaussian_params(X, log_resp, xp=xp)

		self.weights_ /= X.shape[0]
