import chainer.functions as F

from fve_layer.backends.chainer.links.gmm import GMMLayer

class FVELayer(GMMLayer):

	def encode(self, x, use_mask=False, visibility_mask=None, eps=1e-6):
		gamma = self.soft_assignment(x)
		_x, *params = self._expand_params(x)

		_gamma = self.xp.expand_dims(gamma.array, axis=2)
		_mu, _sig, _w = [p.array for p in params]

		"""
			If the GMM component is degenerate and has a null prior, then it
			must have null posterior as well. Hence it is safe to skip it.
			In practice, we skip over it even if the prior is very small;
			if by any chance a feature is assigned to such a mode, then
			its weight would be very high due to the division by
			priors[i_cl] below.

		"""
		# mask out all gammas, that are < eps
		eps_mask = (_gamma >= eps).astype(self.xp.float32)
		_gamma *= eps_mask

		# mask out all weights, that are < eps
		eps_mask = (_w >= eps).astype(self.xp.float32)
		_w = _w * eps_mask

		### Here the computations begin
		_std = F.sqrt(_sig)
		_x_mu_sig = (_x - _mu) / _std

		mask = self.get_mask(x, use_mask, visibility_mask)
		selected = self.xp.zeros(_x_mu_sig.shape, dtype=_x_mu_sig.dtype)
		selected[mask] = 1

		G_mu = _gamma * _x_mu_sig * selected
		G_sig = _gamma * (_x_mu_sig**2 - 1) * selected

		G_mu = F.sum(G_mu, axis=1) / selected.sum(axis=1)
		G_sig = F.sum(G_sig, axis=1) / selected.sum(axis=1)

		_w = self.xp.broadcast_to(self.w, G_mu.shape)
		G_mu /= self.xp.sqrt(_w)
		G_sig /= self.xp.sqrt(2 * _w)

		# 2 * (n, in_size, n_components) -> (n, 2, in_size, n_components)
		res = F.stack([G_mu, G_sig], axis=1)
		# (n, 2, in_size, n_components) -> (n, 2, n_components, in_size)
		res = res.transpose(0, 1, 3, 2)
		# res = F.stack([G_mu], axis=-1)
		# (n, 2, n_components, in_size) -> (n, 2*in_size*n_components)
		res = F.reshape(res, (x.shape[0], -1))
		return res

	def __call__(self, x, use_mask=False, visibility_mask=None):
		_ = super(FVELayer, self).__call__(x, use_mask, visibility_mask)
		return self.encode(x, use_mask, visibility_mask)
