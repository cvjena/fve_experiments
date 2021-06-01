import abc
import chainer
import unittest
import numpy as np

def _as_array(var):
	return var.array if isinstance(var, chainer.Variable) else var

class BaseFVEncodingTest(abc.ABC, unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(BaseFVEncodingTest, self).__init__(*args, **kwargs)
		self.xp = np
		self.dtype = chainer.config.dtype
		self.atol = 1e-1 if self.dtype == chainer.mixed16 else 1e-4
		self.rtol = 1e-1 if self.dtype == chainer.mixed16 else 1e-3

	def assertClose(self, arr0, arr1, msg):
		self.assertTrue(np.allclose(_as_array(arr0), _as_array(arr1), rtol=self.rtol, atol=self.atol),
			f"{msg}:\n{arr0}\n!=\n{arr1}")

	def setUp(self):
		self.n, self.t, self.in_size = 8, 4, 128
		self.n_components = 2
		self.alpha = 0.9

		self.seed = None

		self.rnd = self.xp.random.RandomState(self.seed)

		X = self.rnd.randn(self.n, self.t, self.in_size).astype(self.dtype)
		self.X = chainer.Variable(X, name="TestInput")

		self.init_mu = self.rnd.randn(self.in_size, self.n_components).astype(self.dtype)
		self.init_sig = self.rnd.rand(self.in_size, self.n_components).astype(self.dtype)

		# [0 .. 1] -> [.5 .. 2]
		self.init_sig *= 1.5
		self.init_sig += .5

	@abc.abstractmethod
	def _new_layer(self, layer_cls, init_mu=None, init_sig=None, **kwargs):
		return layer_cls(
			self.in_size,
			self.n_components,
			use_sk_learn_em=True,
			init_mu=self.init_mu if init_mu is None else init_mu,
			init_sig=self.init_sig if init_sig is None else init_sig,
			alpha=self.alpha,
			**kwargs)
