import abc
import chainer
import numpy as np

from chainer import functions as F
from chainer import initializers
from chainer.backends import cuda

from fve_layer.backends.chainer.links.base import BaseEncodingLayer
from fve_layer.common import mixtures
from fve_layer.common import visualization

class GMMMixin(abc.ABC):

	def __init__(self, *args,
		gmm_cls=mixtures.GMM,
		sk_learn_kwargs=dict(
			max_iter=1,
			tol=np.inf,
			reg_covar=1e-2,
		),
		**kwargs):

		super(GMMMixin, self).__init__(*args, **kwargs)

		self.gmm_cls = gmm_cls
		sk_learn_kwargs["reg_covar"] = sk_learn_kwargs.get("reg_covar", self.eps)
		self.sk_learn_kwargs = sk_learn_kwargs
		self.sk_gmm = None

	@abc.abstractmethod
	def set_gmm_params(self, gmm):
		pass

	def new_gmm(self, gmm_cls=None, *args, **kwargs):
		return (gmm_cls or self.gmm_cls)(
			covariance_type="diag",
			n_components=self.n_components,
			**kwargs
		)

	def as_sklearn_gmm(self, gmm_cls=None, **gmm_kwargs):
		gmm = self.new_gmm(gmm_cls=gmm_cls, warm_start=True, **gmm_kwargs)
		self.set_gmm_params(gmm)
		return gmm

	def sample(self, n_samples):
		gmm = self.as_sklearn_gmm(**self.sk_learn_kwargs)
		return gmm.sample(n_samples)

	@property
	def precisions_chol(self):
		"""
			Compute the Cholesky decomposition of the precisions.
			Reference:
				https://github.com/scikit-learn/scikit-learn/blob/0.21.3/sklearn/mixture/gaussian_mixture.py#L288
		"""
		return 1. / self.xp.sqrt(self.sig)

	def plot(self, ax=None, x=None, label=True):
		assert self.in_size == 2, \
			"Plotting is only for 2D mixtures!"
		gmm = self.as_sklearn_gmm()
		ax = visualization.plot_gmm(gmm, X=x, label=label, ax=ax)
		visualization.plot_grad(list(self.params()), ax=ax)

class GMMLayer(GMMMixin, BaseEncodingLayer):

	def __init__(self, in_size, n_components, *,
		init_from_data=False,
		alpha=0.99,
		**kwargs):
		super(GMMLayer, self).__init__(in_size, n_components, **kwargs)

		with self.init_scope():
			self.add_persistent("alpha", alpha)
			self.add_persistent("t", 1)

		self.i = 0
		self.lim = 1
		self.visualization_interval = 100
		self.visualization_folder = "mu_change"

		self._initialized = not init_from_data

	def add_params(self, dtype):

		self.add_persistent("mu",
			np.zeros((self.in_size, self.n_components), dtype))

		self.add_persistent("sig",
			np.zeros((self.in_size, self.n_components), dtype))

		self.add_persistent("w",
			np.zeros((self.n_components), dtype))

	def reset(self):
		self.t = 1 # pragma: no cover

	def init_from_data(self, x, gmm_cls=None):

		if isinstance(x, chainer.Variable):
			data = cuda.to_cpu(x.array)
		else:
			data = cuda.to_cpu(x)

		gmm = self.new_gmm(gmm_cls=gmm_cls)
		self.set_gmm_params(gmm)

		gmm.fit(data.reshape(-1, data.shape[-1]))

		self.mu[:]  = self.xp.array(gmm.means_.T)
		self.sig[:] = self.xp.array(gmm.covariances_.T)
		self.w[:]   = self.xp.array(gmm.weights_)

		self._initialized = True

	def set_gmm_params(self, gmm):
		means_, covariances_, prec_chol_, weights_ = \
			[self.mu.T, self.sig.T, self.precisions_chol.T, self.w]

		gmm.precisions_cholesky_ = prec_chol_
		gmm.covariances_ = covariances_
		gmm.means_= means_
		gmm.weights_= weights_


	def _sk_learn_dist(self, x):
		"""
			Estimate the log Gaussian probability
			Reference:
				https://github.com/scikit-learn/scikit-learn/blob/0.21.3/sklearn/mixture/gaussian_mixture.py#L380
		"""
		n, t, size = x.shape
		_x = x.reshape(-1, size).array
		_mu = self.mu.T
		_precs = 1 / self.sig.T

		res0 = F.sum((_mu ** 2 * _precs), 1)
		res1 = -2. * F.matmul(_x, (_mu * _precs).T)
		res2 = F.matmul(_x ** 2, _precs.T)
		res = F.broadcast_to(res0, res1.shape) + res1 + res2
		return res.reshape(n, t, -1)

	def _log_proba_intern(self, x, use_sk_learn=False):

		if not use_sk_learn:
			return super(GMMLayer, self)._log_proba_intern(x)

		_x, _mu, _sig, _w = self._expand_params(x)
		_dist = self._sk_learn_dist(x)
		prec_chol_ = self.precisions_chol
		# det(precision_chol) is half of det(precision)
		log_det_chol = self.xp.sum(self.xp.log(prec_chol_), axis=0)
		_log_proba = -0.5 * (self.in_size * self._LOG_2PI + _dist) + log_det_chol

		return _log_proba, _w

	def forward(self, x, use_mask=False, visibility_mask=None):
		if chainer.config.train:
			mask = self.get_mask(x,
			                     use_mask=use_mask,
			                     visibility_mask=visibility_mask)
			selected = x[mask]
			selected = selected.reshape(-1, x.shape[-1])
			self.update_parameter(selected)
		return x

	def _ema(self, old, new):
		prev_correction = 1 - (self.alpha ** (self.t-1))
		correction = 1 - (self.alpha ** self.t)

		uncorrected_old = old * prev_correction
		res = self.alpha * uncorrected_old + (1 - self.alpha) * new

		return res / correction

	def get_new_params(self, x):
		if self.sk_gmm is None:
			# self.sk_gmm = self.as_sklearn_gmm(**self.sk_learn_kwargs)
			self.sk_gmm = self.new_gmm(**self.sk_learn_kwargs)

		self.sk_gmm.fit(x)

		new_mu, new_sig, new_w = map(self.xp.array,
			[self.sk_gmm.means_.T, self.sk_gmm.covariances_.T, self.sk_gmm.weights_.T])


		return new_mu, new_sig, new_w

	def update_parameter(self, x):
		if not self._initialized:
			self.init_from_data(x)

		if self.alpha >= 1:
			return #pragma: no cover

		new_mu, new_sig, new_w = self.get_new_params(x)

		self.w[:] = self._ema(self.w, new_w)
		self.mu[:]  = self._ema(self.mu, new_mu)
		self.sig[:] = self._ema(self.sig, new_sig)
		self.t += 1

		self.sig = self.xp.maximum(self.sig, self.eps)

		# self.i += 1
		# if (self.i-1) % self.visualization_interval == 0:
		# 	self.__visualize(x, gamma, new_mu, None, new_w)
