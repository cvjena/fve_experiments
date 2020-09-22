import chainer
import numpy as np

from scipy.stats import multivariate_normal as mvn

from fve_layer.backends.chainer.links import GMMLayer
from tests.base import BaseFVEncodingTest

class GMMLayerTest(BaseFVEncodingTest):

	def _new_layer(self, *args, **kwargs):
		return super(GMMLayerTest, self)._new_layer(layer_cls=GMMLayer)

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
