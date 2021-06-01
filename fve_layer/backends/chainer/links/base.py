import abc
import chainer
import numpy as np

from chainer import initializers
from chainer import link
from chainer import functions as F
from chainer.backends import cuda
from functools import wraps

def promote_x_dtype(method):

	def cast(arr, dtype, xp=np):
		caster = F.cast if isinstance(arr, chainer.Variable) else xp.array
		return caster(arr, dtype)

	@wraps(method)
	def inner(self, x, *args, **kwargs):
		x_dtype = x.dtype
		interm_dtype = np.promote_types(x_dtype, self.mu.dtype)

		if x_dtype == interm_dtype:
			return method(self, x, *args, **kwargs)

		x = cast(x, interm_dtype, xp=self.xp)
		y = method(self, x, *args, **kwargs)

		if isinstance(y, tuple):
			return (cast(_y, x_dtype, xp=self.xp) for _y in y)

		elif isinstance(y, list):
			return [cast(_y, x_dtype, xp=self.xp) for _y in y]

		return cast(y, x_dtype, xp=self.xp)

	return inner


class BaseEncodingLayer(link.Link, abc.ABC):
	_LOG_2PI = np.log(2 * np.pi)

	def __init__(self, in_size, n_components, *,
		init_mu=None,
		init_sig=1,
		eps=1e-2,
		dtype=chainer.get_dtype(map_mixed16=np.float32),
		**kwargs):
		super(BaseEncodingLayer, self).__init__()

		self.n_components = n_components
		self.in_size = in_size

		with self.init_scope():
			self.add_persistent("eps", eps)

		self._init_initializers(init_mu, init_sig, dtype)

		with self.init_scope():
			self.add_params(dtype)
			self.init_params()

	@property
	def printable_specs(self):
		specs = [
			('in_size', self.in_size),
			('n_components', self.n_components),
			('eps', self.eps),
		]
		for spec in specs:
			yield spec

	@abc.abstractmethod
	def add_params(self, dtype):
		pass

	def _init_initializers(self, init_mu, init_sig, dtype):

		if init_mu is None:
			self.init_mu = initializers.Uniform(scale=1, dtype=dtype)

		elif isinstance(init_mu, (int, float)):
			self.init_mu = initializers.Uniform(scale=init_mu, dtype=dtype)

		elif isinstance(init_mu, (np.ndarray, cuda.ndarray)):
			self.init_mu = initializers.Constant(init_mu, dtype=dtype)

		elif not isinstance(init_mu, chainer.initializer.Initializer):
			raise ValueError(
				"\"init_mu\" should be either omited, be an instance of " + \
				"chainer.initializer.Initializer or a numpy/cupy array!"
			)

		self.init_sig = initializers.Constant(init_sig, dtype=dtype)
		self.init_w = initializers.Constant(1 / self.n_components, dtype=dtype)


	def init_params(self):
		self.init_mu(self.mu)
		self.init_sig(self.sig)
		self.init_w(self.w)

	def _check_input(self, x):
		assert x.ndim == 3, \
			"input should have following dimensions: (batch_size, n_features, feature_size)"
		n, t, in_size = x.shape
		assert in_size == self.in_size, \
			"feature size of the input does not match input size: ({} != {})! ".format(
				in_size, self.in_size)
		return n, t

	def _expand_params(self, x):
		n, t = self._check_input(x)
		shape = (n, t, self.in_size, self.n_components)
		shape2 = (n, t, self.n_components)

		_x = F.broadcast_to(F.expand_dims(x, -1), shape)

		_params = [(self.mu, shape), (self.sig, shape), (self.w, shape2)]
		_ps = []
		for p, s in _params:
			_p = F.expand_dims(F.expand_dims(p, 0), 0)
			# print(p.shape, _p.shape, s, sep=" -> ")
			_ps.append(F.broadcast_to(_p, s))
		_mu, _sig, _w = _ps
		# _mu, _sig, _w = [
		# 	F.broadcast_to(F.expand_dims(F.expand_dims(p, 0), 0), s)
		# 		for p, s in _params]

		return _x, _mu, _sig, _w

	@promote_x_dtype
	def soft_assignment(self, x):
		return F.exp(self.log_soft_assignment(x))

	def log_soft_assignment(self, x):

		_log_proba, _w = self.log_proba(x, weighted=False)
		_log_wu = _log_proba + F.log(_w)

		_log_wu_sum = F.logsumexp(_log_wu, axis=-1)
		_log_wu_sum = F.expand_dims(_log_wu_sum, axis=-1)
		_log_wu_sum = F.broadcast_to(_log_wu_sum, _log_wu.shape)

		return _log_wu - _log_wu_sum

	@promote_x_dtype
	def _dist(self, x, *, return_weights=True):
		"""
			computes squared Mahalanobis distance
			(in our case it is the standartized Euclidean distance)
		"""
		_x, _mu, _sig, _w = self._expand_params(x)

		_dist = F.sum((_x - _mu)**2 / _sig, axis=2)

		return (_dist, _w) if return_weights else _dist

	def mahalanobis_dist(self, x):
		_dist = self._dist(x, return_weights=False)
		return F.sqrt(_dist)

	@promote_x_dtype
	def _log_proba_intern(self, x):
		_dist, _w = self._dist(x, return_weights=True)

		# normalize with (2*pi)^k and det(sig) = prod(diagonal_sig)
		log_det = F.sum(F.log(self.sig), axis=0)
		_log_proba = -0.5 * (self.in_size * self._LOG_2PI + _dist + log_det)

		return _log_proba, _w

	def log_proba(self, x, weighted=False, *args, **kwargs):

		_log_proba, _w = self._log_proba_intern(x, *args, **kwargs)

		if weighted:
			_log_wu = _log_proba + F.log(_w)
			_log_proba = F.logsumexp(_log_wu, axis=-1)

		return _log_proba, _w

	def proba(self, *args, **kwargs):
		_log_proba, _w = self.log_proba(*args, **kwargs)
		return F.exp(_log_proba), _w

	def get_mask(self, x, use_mask, visibility_mask=None):
		if not use_mask: return Ellipsis
		_feats = x.array if hasattr(x, "array") else x
		_feat_lens = self.xp.sqrt(self.xp.sum(_feats**2, axis=2))


		if visibility_mask is None:
			_mean_feat_lens = _feat_lens.mean(axis=1, keepdims=True)
			selected = _feat_lens >= _mean_feat_lens

		else:

			if 0 in visibility_mask.sum(axis=1):
				raise RuntimeError("Selection mask contains not selected samples!")

			_mean_feat_lens = (_feat_lens * visibility_mask).sum(axis=1, keepdims=True)
			_n_visible_feats = visibility_mask.sum(axis=1, keepdims=True).astype(chainer.config.dtype)
			_mean_feat_lens /= _n_visible_feats

			selected = _feat_lens >= _mean_feat_lens
			selected = self.xp.logical_and(selected, visibility_mask)

		return self.xp.where(selected)
