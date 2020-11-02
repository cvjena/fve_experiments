import chainer
import numpy as np

from chainer.initializers import HeNormal
from chainer.serializers import load_npz
from chainer_addons.links import PoolingType

class ModelWrapper(chainer.Chain):
	class meta(object):
		input_size = 224
		feature_size = None
		classifier_layers = ["fc"]
		conv_map_layer = "stage4"
		feature_layer = "final_pool"

	def __init__(self, model, pooling=PoolingType.Default):
		super(ModelWrapper, self).__init__()
		self.__class__.__name__ = model.__class__.__name__

		with self.init_scope():
			self.wrapped = model
			self.pool = PoolingType.new(pooling)
			delattr(self.wrapped.features, self.meta.feature_layer)

		self.meta.feature_size = self.clf_layer.W.shape[-1]

	@property
	def clf_layer_name(self):
		return "output/fc"

	@property
	def clf_layer(self):
		return self.wrapped.output.fc


	def load_for_finetune(self, weights, n_classes, *, path="", strict=False, headless=False, **kwargs):
		"""
			The weights should be pre-trained on a bigger
			dataset (eg. ImageNet). The classification layer is
			reinitialized after all other weights are loaded
		"""
		self.load(weights, path=path, strict=strict, headless=headless)
		self.reinitialize_clf(n_classes, **kwargs)

	def load_for_inference(self, weights, n_classes, *, path="", strict=False, headless=False, **kwargs):
		"""
			In this use case we are loading already fine-tuned
			weights. This means, we need to reinitialize the
			classification layer first and then load the weights.
		"""
		self.reinitialize_clf(n_classes, **kwargs)
		self.load(weights, path=path, strict=strict, headless=headless)

	def load(self, weights, *, path="", strict=False, headless=False):
		if weights not in [None, "auto"]:
			ignore_names = None
			if headless:
				ignore_names = lambda name: name.startswith(path + self.clf_layer_name)

			load_npz(weights, self.wrapped,
				path=path, strict=strict,
				ignore_names=ignore_names)

	def reinitialize_clf(self, n_classes,
		feat_size=None, initializer=None):
		if initializer is None or not callable(initializer):
			initializer = HeNormal(scale=1.0)

		clf_layer = self.clf_layer

		w_shape = (n_classes, feat_size or clf_layer.W.shape[1])
		dtype = clf_layer.W.dtype
		clf_layer.W.data = np.zeros(w_shape, dtype=dtype)
		clf_layer.b.data = np.zeros(w_shape[0], dtype=dtype)
		initializer(clf_layer.W.data)


	def __call__(self, X, layer_name=None):
		if layer_name is None:
			res = self.wrapped(X)

		elif layer_name == self.meta.conv_map_layer:
			res = self.wrapped.features(X)

		elif layer_name == self.feature_layer:
			conv = self.wrapped.features(X)
			res = self.pool(conv)

		elif layer_name == self.clf_layer_name:
			conv = self.wrapped.features(X)
			feat = self.pool(conv)
			res = self.wrapped.output(feat)

		else:
			raise ValueError(f"Dont know how to compute \"{layer_name}\"!")

		return res
