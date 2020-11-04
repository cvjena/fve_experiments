import chainer
import numpy as np

from scipy.stats import multivariate_normal as mvn

from fve_layer.backends.chainer.links import GMMLayer
from fve_layer.common.mixtures import BayesianGMM
from tests.base import BaseFVEncodingTest
from tests.base import _as_array

class GMMLayerTest(BaseFVEncodingTest):

	def _new_layer(self, *args, **kwargs):
		return super(GMMLayerTest, self)._new_layer(layer_cls=GMMLayer, *args, **kwargs)


	def test_initialization(self):
		layer = self._new_layer(init_mu=10)
		self.assertTrue(layer.mu.min() >= -10)
		self.assertTrue(layer.mu.max() <=  10)

		self.init_mu = None
		layer = self._new_layer()
		self.assertTrue(layer.mu.min() >= -1)
		self.assertTrue(layer.mu.max() <=  1)


		with self.assertRaises(ValueError):
			self.init_mu = "None"
			self._new_layer()

	def test_masks(self):

		layer = self._new_layer()
		res = layer(self.X, use_mask=True)


		layer = self._new_layer()
		vis_mask = layer.xp.ones(self.X.shape[:-1], dtype=bool)
		idxs = layer.xp.random.randint(self.t, size=self.n)
		vis_mask[np.arange(self.n), idxs] = 0
		res = layer(self.X, use_mask=True, visibility_mask=vis_mask)

		with self.assertRaises(RuntimeError):
			layer = self._new_layer()
			vis_mask = layer.xp.zeros(self.X.shape[:-1], dtype=bool)
			res = layer(self.X, use_mask=True, visibility_mask=vis_mask)

	def test_init_from_data(self):

		layer = self._new_layer(init_from_data=True)
		res = layer(self.X)

		layer = self._new_layer(init_from_data=True)
		res = layer(_as_array(self.X))

	def test_sklearn_dist(self):
		layer = self._new_layer()
		res0, _ = layer.log_proba(self.X, use_sk_learn=False)
		res1, _ = layer.log_proba(self.X, use_sk_learn=True)

		self.assertClose(res0, res1,
			"sklearn results in a different result!")

	def test_gpu(self):
		layer = self._new_layer()
		device = chainer.backends.cuda.get_device_from_id(0)
		layer.to_device(device)
		self.X.to_device(device)

		res = layer(self.X)

	def test_output(self):
		layer = self._new_layer()

		res = layer(self.X)

		self.assertIs(res, self.X,
			"Output should be the input!")

	def test_update(self):
		layer = self._new_layer()
		params = (layer.mu, layer.sig, layer.w)

		params0 = [np.copy(p) for p in params]
		with chainer.using_config("train", False):
			y0 = layer(self.X)
		params1 = [np.copy(p) for p in params]

		for p0, p1 in zip(params0, params1):
			self.assertTrue(np.all(p0 == p1),
				"Params should not be updated when not training!")


		params0 = [np.copy(p) for p in params]
		with chainer.using_config("train", True):
			y0 = layer(self.X)
		params1 = [np.copy(p) for p in params]

		for p0, p1 in zip(params0, params1):
			self.assertTrue(np.all(p0 != p1),
				"Params should be updated when training!")

	def test_assignment_shape(self):
		layer = self._new_layer()

		gamma = layer.soft_assignment(self.X).array
		correct_shape = (self.n, self.t, self.n_components)
		self.assertEqual(gamma.shape, correct_shape,
			"Shape of the soft assignment is not correct!")

	def test_assignment_sum(self):
		layer = self._new_layer()

		gamma = layer.soft_assignment(self.X).array
		gamma_sum = gamma.sum(axis=-1)
		self.assertClose(gamma_sum, 1,
			"Sum of the soft assignment should be always equal to 1, but was")

	def test_assignment(self):
		layer = self._new_layer()
		correct_shape = (self.n, self.t, self.n_components)
		gamma = layer.soft_assignment(self.X).array

		gmm = layer.as_sklearn_gmm()
		x = self.X.reshape(-1, self.in_size).array
		ref_gamma = gmm.predict_proba(x).reshape(correct_shape)

		self.assertClose(gamma, ref_gamma,
			"Soft assignment is not similar to reference value from sklearn")


	def test_log_assignment(self):
		layer = self._new_layer()
		correct_shape = (self.n, self.t, self.n_components)
		log_gamma = layer.log_soft_assignment(self.X).array

		gmm = layer.as_sklearn_gmm()
		x = self.X.reshape(-1, self.in_size).array
		_, log_ref_gamma = gmm._estimate_log_prob_resp(x)
		log_ref_gamma = log_ref_gamma.reshape(correct_shape)

		self.assertClose(log_gamma, log_ref_gamma,
			"Log soft assignment is not similar to reference value from sklearn")

	def test_log_proba(self):

		layer = self._new_layer()

		params = mean, var, _ = (layer.mu, layer.sig, layer.w)
		mean, var = mean.T, var.T

		log_ps2, _ = layer.log_proba(self.X, weighted=False)
		ps2, _ = layer.proba(self.X, weighted=False)


		for i, x in enumerate(self.X.reshape(-1, self.in_size).array):
			n, t = np.unravel_index(i, (self.n, self.t))

			for component in range(self.n_components):
				log_p1 = mvn.logpdf(x, mean[component], var[component]).astype(self.dtype)
				log_p2 = log_ps2[n, t, component].array

				self.assertClose(log_p1, log_p2,
					f"{[n,t,component]}: Log-Likelihood was not the same")

				p1 = mvn.pdf(x, mean[component], var[component]).astype(self.dtype)
				p2 = ps2[n, t, component].array

				self.assertClose(p1, p2,
					f"{[n,t,component]}: Likelihood was not the same")
