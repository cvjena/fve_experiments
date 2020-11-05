import abc
import chainer
import numpy as np

from chainer import functions as F
from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import cygmm
from cyvlfeat.gmm import gmm

from fve_layer.backends.chainer.links import FVELayer
from fve_layer.backends.chainer.links import FVELayer_noEM
from tests.base import BaseFVEncodingTest
from tests.base import _as_array

class BaseFVELayerTest(BaseFVEncodingTest):

	@abc.abstractmethod
	def _new_layer(self, *args, **kwargs):
		return super(BaseFVELayerTest, self)._new_layer(*args, **kwargs)

	def test_output_shape(self):
		layer = self._new_layer()

		with chainer.using_config("train", False):
			output = layer(self.X)

		output_shape = (self.n, 2 * self.n_components * self.in_size)
		self.assertEqual(output.shape, output_shape,
			"Output shape was not correct!")

	def test_output(self):
		layer = self._new_layer()

		mean, var, w = map(_as_array, [layer.mu, layer.sig, layer.w])

		with chainer.using_config("train", False):
			output = layer(self.X).array

		x = self.X.array.astype(np.float32)

		# we need to convert the array in Fortran order, but still remain the dimensions,
		# since the python API awaits <dimensions>x<components> arrays,
		# but they are indexed with "i_cl*dimension + dim" in the C-Code

		params = (
			mean.T.copy(),
			var.T.copy(),
			w.copy(),
		)
		ref = [fisher(_x, *params,
					normalized=False,
					square_root=False,
					improved=False,
					fast=False,
					verbose=False,
			) for _x in x]

		ref = np.stack(ref)

		self.assertEqual(output.shape, ref.shape,
			"Output shape was not equal to ref shape!")

		self.assertClose(output, ref,
			"Output was not similar to reference")


	def test_cygmm(self):
		layer = self._new_layer()
		mean, var, w = map(_as_array, [layer.mu, layer.sig, layer.w])

		x = self.X.array.astype(np.float32)
		gamma = layer.soft_assignment(x).array
		log_proba, _ = layer.log_proba(x, weighted=True)
		log_proba = log_proba.array

		for i, _x in enumerate(x):
			cy_mean, cy_var, cy_w, LL, cy_gamma = cygmm.cy_gmm(
				_x,
				self.n_components,
				0, # max_iterations
				"custom".encode("utf8"), # init_mode
				1, # n_repitions
				0, # verbose
				covariance_bound=None,

				init_means=mean.T.copy(),
				init_covars=var.T.copy(),
				init_priors=w.copy(),
			)

			self.assertClose(gamma[i], cy_gamma,
				f"[{i}] Soft assignment was not similar to reference (vlfeat)")

			self.assertClose(float(log_proba[i].sum()), LL,
				f"[{i}] Log-likelihood was not similar to reference (vlfeat)")


	def test_gap_init(self):
		self.n_components = 1
		layer = self._new_layer(init_mu=0, init_sig=1)
		_x = self.X.array
		gap_output = self.xp.mean(_x, axis=1)
		ref_sig_output = self.xp.mean(_x**2 - 1, axis=1) / self.xp.sqrt(2)

		with chainer.using_config("train", False):
			output = layer(self.X).array

		mu_output, sig_output = output[:, :self.in_size], output[:, self.in_size:]

		self.assertClose(mu_output, gap_output,
			"mu-Part of FVE should be equal to GAP!")

		self.assertClose(sig_output, ref_sig_output,
			"sig-Part of FVE should be equal to reference!")


class FVELayerTest(BaseFVELayerTest):

	def _new_layer(self, *args, **kwargs):
		return super(FVELayerTest, self)._new_layer(layer_cls=FVELayer, *args, **kwargs)

class FVELayer_noEMTest(BaseFVELayerTest):

	def _new_layer(self, *args, **kwargs):
		return super(FVELayer_noEMTest, self)._new_layer(layer_cls=FVELayer_noEM, *args, **kwargs)


	def test_gradients(self):

		layer = self._new_layer()
		layer.cleargrads()

		for param in layer.params():
			self.assertIsNone(param.grad)

		with chainer.using_config("train", True), chainer.force_backprop_mode():
			output = layer(self.X)

			# from chainer.computational_graph import build_computational_graph as bcg
			# import graphviz
			# g = bcg([output])
			# graphviz.Source(g.dump()).render(view=True)

			output.grad = layer.xp.ones_like(output.array)

			output.backward()

		for param in layer.params():
			self.assertIsNotNone(param.grad)
