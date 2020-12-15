import abc
import chainer

from chainer import functions as F
from chainer import links as L

from fve_layer.backends.chainer.links.gmm import GMMLayer
from fve_layer.backends.chainer.links.gmm import GMMMixin
from fve_layer.backends.chainer.links.base import BaseEncodingLayer


class FVEMixin(abc.ABC):

	def encode(self, x, use_mask=False, visibility_mask=None, eps=1e-6):
		gamma = self.soft_assignment(x)
		xp = self.xp
		_x, *params = self._expand_params(x)

		_gamma = xp.expand_dims(gamma.array, axis=2)
		_mu, _sig, _w = [p for p in params]

		"""
			If the GMM component is degenerate and has a null prior, then it
			must have null posterior as well. Hence it is safe to skip it.
			In practice, we skip over it even if the prior is very small;
			if by any chance a feature is assigned to such a mode, then
			its weight would be very high due to the division by
			priors[i_cl] below.

		"""
		# mask out all gammas, that are < eps
		eps_mask = (_gamma >= eps).astype(xp.float32)
		_gamma *= eps_mask

		# mask out all weights, that are < eps
		eps_mask = (_w.array >= eps).astype(xp.float32)
		_w = _w * eps_mask

		### Here the computations begin
		_std = F.sqrt(_sig)
		_x_mu_sig = (_x - _mu) / _std

		mask = self.get_mask(x, use_mask, visibility_mask)
		selected = xp.zeros(_x_mu_sig.shape, dtype=_x_mu_sig.dtype)
		selected[mask] = 1

		G_mu = _gamma * _x_mu_sig * selected
		G_sig = _gamma * (_x_mu_sig**2 - 1) * selected

		"""
			Here we are not so sure about the normalization factor.
			In [1] the factor is 1 / sqrt(T) (Eqs. 10, 11, 13, 4),
			but in [2] the factor is 1 / T (Eqs. 7, 8).

			Actually, this has no effect on the final classification,
			but is still a mismatch in the literature.

			In this code, we stick to the version of [2], since it
			fits the results computed by cyvlfeat.

			---------------------------------------------------------------------
			[1] - Fisher Kernels on Visual Vocabularies for Image Categorization
			(https://ieeexplore.ieee.org/document/4270291)
			[2] - Improving the Fisher Kernel for Large-Scale Image Classification
			(https://link.springer.com/chapter/10.1007/978-3-642-15561-1_11)
		"""
		# Version 1:
		# G_mu = F.sum(G_mu, axis=1) / xp.sqrt(selected.sum(axis=1))
		# G_sig = F.sum(G_sig, axis=1) / xp.sqrt(selected.sum(axis=1))
		# Version 2:
		G_mu = F.sum(G_mu, axis=1) / selected.sum(axis=1)
		G_sig = F.sum(G_sig, axis=1) / selected.sum(axis=1)

		_w = F.broadcast_to(self.w, G_mu.shape)
		G_mu /= F.sqrt(_w)
		G_sig /= F.sqrt(2 * _w)

		# 2 * (n, in_size, n_components) -> (n, 2, in_size, n_components)
		res = F.stack([G_mu, G_sig], axis=1)
		# (n, 2, in_size, n_components) -> (n, 2, n_components, in_size)
		res = res.transpose(0, 1, 3, 2)
		# res = F.stack([G_mu], axis=-1)
		# (n, 2, n_components, in_size) -> (n, 2*in_size*n_components)
		res = F.reshape(res, (x.shape[0], -1))
		return res

class FVELayer(FVEMixin, GMMLayer):

	def forward(self, x, use_mask=False, visibility_mask=None):
		_ = super(FVELayer, self).forward(x, use_mask, visibility_mask)
		return self.encode(x, use_mask, visibility_mask)

class FVELayer_noEM(FVEMixin, GMMMixin, BaseEncodingLayer):

	def _init_initializers(self, init_mu, init_sig, dtype):
		init_sig = self.xp.log(init_sig - self.eps)
		super(FVELayer_noEM, self)._init_initializers(init_mu, init_sig, dtype)


	def add_params(self, dtype):

		self.mu = chainer.Parameter(
			initializer=self.init_mu,
			shape=(self.in_size, self.n_components),
			name="mu")

		self._sig = chainer.Parameter(
			initializer=self.init_sig,
			shape=(self.in_size, self.n_components),
			name="sig")

		self._w = chainer.Parameter(
			initializer=self.init_w,
			shape=(self.n_components,),
			name="w")

	@property
	def w(self):
		w_sigmoid = F.sigmoid(self._w)
		return w_sigmoid / F.sum(w_sigmoid)

	@w.setter
	def w(self, param):
		self._w = param

	@property
	def sig(self):
		return self.eps + F.exp(self._sig)

	def init_params(self):
		pass

	def set_gmm_params(self, gmm):

		gmm.precisions_cholesky_ = self.precisions_chol.array.T
		gmm.covariances_ = self.sig.array.T
		gmm.means_= self.mu.array.T
		gmm.weights_= self.w.array

	@property
	def precisions_chol(self):
		return 1. / F.sqrt(self.sig)

	def forward(self, x, use_mask=False, visibility_mask=None):
		return self.encode(x, use_mask, visibility_mask)
