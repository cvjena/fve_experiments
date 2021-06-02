import abc
import chainer
import numpy as np
import warnings

# warnings.warn("[FVE-Layer] Currently, only chainer>=6.7.0 is supported!")

from chainer import functions as F
from chainer.backends import cuda

from sklearn.utils import check_random_state


_LOG_2PI = np.log(2 * np.pi)

def _kernel_e_step(X, means, cov, ws, xp=cuda.cupy):

	t, size = X.shape
	n_comp, size = means.shape
	log_det = xp.sum(xp.log(1. / xp.sqrt(cov)), axis=1)


	log_prob = xp.zeros((t, n_comp), dtype=X.dtype)
	log_prob_kernel = cuda.elementwise(
		name="gmm_log_prob",
		in_params="raw T X, T means, T cov, int32 t, int32 size, int32 n_comp",
		out_params="raw T log_prob",
		operation="""
			int ni = i / size; /* component idx */
			int fi = i % size; /* feature idx */
			int j = 0;
			int xi = 0;

			for ( /* part idx */ int ti = 0; ti < t; ti++) {
				int j = ti * n_comp + ni;
				int xi = ti * size + fi;

				/* printf( "%d, %d, %d -> %d, %d \\n", ti, ni, fi, j, xi); */
				atomicAdd( &log_prob[j], powf(X[xi] - means, 2) / cov);
			}
		"""
	)

	log_prob_kernel(X, means, cov, t, size, n_comp, log_prob)

	weighting_kernel = cuda.elementwise(
		name="gmm_weighted_prob",
		in_params="T log_prob, T ws, T log_det, int32 size",
		out_params="T weighted_log_prob",
		operation="weighted_log_prob = -0.5 * (size * LOG_2PI + log_prob) + log_det + log(ws);",
		preamble="#define LOG_2PI log(2 * 3.14159265359)",
	)
	log_prob = weighting_kernel(log_prob, ws, log_det, size)

	norm_kernel = cuda.elementwise(
		name="gmm_norm_log_prob",
		in_params="T log_prob, raw T norm, int32 n_comp",
		out_params="T norm_log_prob",
		operation="norm_log_prob = log_prob - norm[i/n_comp];",
	)

	log_prob_norm = F.logsumexp(log_prob, axis=1).array
	log_resp = norm_kernel(log_prob, log_prob_norm, n_comp)

	return xp.mean(log_prob_norm), log_resp

def _basic_e_step(X, means, cov, ws, xp=np):

	n_features = X.shape[1]

	log_det = xp.sum(xp.log(1. / xp.sqrt(cov)), axis=1)
	precisions = 1. / cov

	res0 = xp.sum((means ** 2 * precisions), 1)
	res1 = -2. * xp.dot(X, (means * precisions).T)
	res2 = xp.dot(X ** 2, precisions.T)

	log_prob = res0 + res1 + res2
	log_prob = -.5 * (n_features * _LOG_2PI + log_prob) + log_det

	weighted_log_prob = log_prob + xp.log(ws)

	log_prob_norm = F.logsumexp(weighted_log_prob, axis=1).array

	log_resp = weighted_log_prob - log_prob_norm[:, None]

	return xp.mean(log_prob_norm), log_resp


class GPUMixin(abc.ABC):

	def xp_from_array(self, X):
		_x = getattr(X, "array", X)

		return chainer.backend.get_array_module(_x)


	def _transform_X(self, X):
		X = X.reshape(-1, X.shape[-1])
		xp = self.xp_from_array(X)
		_x = getattr(X, "array", X)
		return _x, xp

	def _initialize_parameters(self, X, random_state):
		super(GPUMixin, self)._initialize_parameters(cuda.to_cpu(X), random_state)
		xp = self.xp_from_array(X)
		if xp != np:
			self.means_ = xp.array(self.means_, dtype=X.dtype)
			self.covariances_ = xp.array(self.covariances_, dtype=X.dtype)
			self.weights_ = xp.array(self.weights_, dtype=X.dtype)
			self.precisions_cholesky_ = xp.array(self.precisions_cholesky_, dtype=X.dtype)


	def fit(self, X, y=None):
		X, xp = self._transform_X(X)
		self._check_initial_parameters(X)

		random_state = check_random_state(self.random_state)
		attrs = ["means_", "covariances_", "weights_"]
		if not all([hasattr(self, attr) for attr in attrs]):
			self._initialize_parameters(X, random_state)

		for n_iter in range(1, self.max_iter + 1):
			log_prob_norm, log_resp = self._e_step(X, xp=xp)
			self._m_step(X, log_resp, xp=xp)

	def _e_step(self, X, xp=np, use_kernel=True):
		""" E step.
			Copied from sklearn/mixture/base.py
		"""
		if xp != np and use_kernel:
			e_step_impl = _kernel_e_step
		else:
			e_step_impl = _basic_e_step

		return e_step_impl(X, self.means_, self.covariances_, self.weights_, xp=xp)

	@abc.abstractmethod
	def _m_step(self, *args, **kwargs):
		raise NotImplementedError()

	def _gaussian_params(self, X, log_resp, xp):

		resp = xp.exp(log_resp)
		nk = resp.sum(axis=0) + 10 * xp.finfo(resp.dtype).eps
		means = xp.dot(resp.T, X) / nk[:, None]

		avg_X2 = xp.dot(resp.T, X ** 2) / nk[:, None]
		avg_means2 = means ** 2
		avg_X_means = means * xp.dot(resp.T, X) / nk[:, None]
		covariances = avg_X2 - 2 * avg_X_means + avg_means2

		covariances = xp.maximum(covariances, self.reg_covar)

		return nk, means, covariances

	def sample(self, n_samples=1):
		_save = self.means_, self.covariances_, self.weights_

		self.means_, self.covariances_, self.weights_ = map(cuda.to_cpu, _save)
		res = super(GPUMixin, self).sample(n_samples)
		self.means_, self.covariances_, self.weights_ = _save
		return res

# testing the runtimes
if __name__ == '__main__':
	import time
	from functools import partial

	n_iter, warm_up = 10000, 200
	def bench(func, *args, **kwargs):
		global n_iter, warm_up

		for _ in range(warm_up):
			res = func(*args, **kwargs)

		t0 = time.time()

		for _ in range(n_iter):
			res = func(*args, **kwargs)

		t0 = time.time() - t0
		func_name = func.func.__name__ if isinstance(func, partial) else func.__name__
		print(f"{func_name} took {t0:.3f}s for {n_iter} runs")
		return res

	T, N_COMP, SIZE = 64, 10, 32
	dtype, xp = chainer.config.dtype, cuda.cupy

	X = xp.random.randn(T, SIZE).astype(dtype)
	mu = xp.random.rand(N_COMP, SIZE).astype(dtype)
	sig = xp.random.rand(N_COMP, SIZE).astype(dtype) + 1
	ws = xp.ones(N_COMP).astype(dtype) / N_COMP

	logL0, log_gammas0 = bench(_kernel_e_step, X, mu, sig, ws, xp=xp)

	logL1, log_gammas1 = bench(_basic_e_step, X, mu, sig, ws, xp=xp)

	assert xp.allclose(log_gammas0, log_gammas1, atol=1e-5), \
		f"Were not equal: \n{log_gammas0} !=\n{log_gammas1}"

	assert xp.allclose(logL0, logL1, atol=1e-5), \
		f"Were not equal: \n{logL0} !=\n{logL1}\n"
