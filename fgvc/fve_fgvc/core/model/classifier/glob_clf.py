import chainer
import chainer.functions as F
import chainer.links as L

from cvmodelz import classifiers
from functools import partial

from fve_fgvc import utils
from fve_fgvc.core.model.classifier.base import BaseFVEClassifier

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

class GlobalClassifier(BaseFVEClassifier, classifiers.Classifier):

	def loss(self, preds, aux_preds=None, *, y) -> chainer.Variable:

		_loss = partial(self.model.loss, gt=y, loss_func=self.loss_func)
		loss = _loss(preds)

		if aux_preds is None:
			return loss

		aux_loss = _loss(aux_preds)
		self.report(aux_loss=aux_loss)
		return self.aux_lambda * aux_loss + (1 - self.aux_lambda) * loss

	def evaluations(self, preds, aux_preds=None, *, y) -> dict:
		if aux_preds is None:
			return dict(accu=F.accuracy(preds, y))
		else:
			return dict(accu=F.accuracy(preds, y), aux_accu=F.accuracy(aux_preds, y))

	def predict_aux(self, convs):
		assert convs.ndim == 4, \
			f"Malformed aux input: {convs.shape=}"

		# (N, C, H, W) -> (N, C)
		feats = self.model.pool(convs)
		return self.aux_clf(feats)

	@utils.tuple_return
	def encode(self, convs):
		""" Implements the encoding of 4D conv maps.
			In case of missing FVELayer, only model's pooling is applied.
			For the FVELayer, the conv maps are extended to 5D and FVE is performed.
		"""
		assert convs.ndim == 4, \
			f"Malformed encoding input: {convs.shape=}"

		if self.fve_layer is None:
			# N x C x H x W -> N x C
			return self.model.pool(convs)

		# expand to 5D: N x C x H x W -> N x T x C x H x W
		convs = F.expand_dims(convs, axis=1)
		return self.fve_encode(convs)

	@utils.tuple_return
	def predict(self, feats):
		return self.model.clf_layer(feats)


class SelfAttentionClassifier(GlobalClassifier):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		with self.init_scope():
			self._init_attention(k=8)

	def _init_attention(self, k=8):
		in_size = self.model.meta.n_conv_maps
		out_size = int(in_size / 8)

		self.query = L.Convolution2D(in_size, out_size, ksize=1)
		self.key = L.Convolution2D(in_size, out_size, ksize=1)
		self.value = L.Convolution2D(in_size, out_size, ksize=1)
		self.out = L.Convolution2D(out_size, in_size, ksize=1)

		self.gamma = chainer.Parameter(0.0)
		self.gamma.initialize(None)


	def self_attention(self, convs):
		assert convs.ndim == 4, \
			f"invalid input shape: {convs.shape}"
		N, C, H, W = convs.shape

		query = self.query(convs).reshape(N, -1, H*W)
		key = self.key(convs).reshape(N, -1, H*W)
		value = self.value(convs).reshape(N, -1, H*W)

		# energy.shape = N x (H * W) x (H * W)
		energy = F.matmul(query, key, transa=True)

		# attention.shape = N x (H * W) x (H * W)
		attention = F.softmax(energy, axis=-1)

		out = F.matmul(value, attention, transb=True)
		out = self.out(out.reshape(N, -1, H, W))

		self.report(gamma=self.gamma)

		self._visualize(convs, self.gamma*out)

		return self.gamma * out + convs


	@utils.tuple_return
	def encode(self, convs):

		att_convs = self.self_attention(convs)
		return super().encode(att_convs)

	def extract(self, X, *args, **kwargs):
		self._cur_X = X
		return super().extract(X, *args, **kwargs)

	def _visualize(self, convs, atts):

		_to_cpu = lambda arr: chainer.cuda.to_cpu(chainer.as_array(arr))

		def _normalize(arr, chan_axis=0, channel_wise=False):
			arr = _to_cpu(arr)

			if channel_wise:
				non_chan_axis = tuple([i for i in range(arr.ndim) if i != chan_axis])
				arr -= arr.min(axis=non_chan_axis, keepdims=True)

				_max_vals = arr.max(axis=non_chan_axis, keepdims=True)
				mask = (_max_vals != 0).squeeze()
				if mask.any():
					arr[mask] /= _max_vals[mask]
					arr = arr.sum(axis=chan_axis) / mask.sum()
				else:
					arr = arr.mean(axis=chan_axis)

			else:
				arr = np.abs(arr).mean(axis=chan_axis)
				arr -= arr.min()
				if arr.max() != 0:
					arr /= arr.max()

			return arr

		def _prepare_back(x):
			x = _to_cpu(x)
			x = (x + 1) / 2
			return x.transpose(1,2,0)

		cpu_convs = _to_cpu(convs)
		SOFT_ASSIGNMENT = True

		###### CAM #####
		with chainer.using_config("train", False), chainer.no_backprop_mode():
			_feats = super().encode(convs+atts)
			_preds = self.predict(*_feats)
			_pred = _to_cpu(F.softmax(_preds[0]))
			_W = _to_cpu(self.model.clf_layer.W)
			_classes = np.argsort(-_pred, axis=-1)

			if SOFT_ASSIGNMENT:
				# NxCLS @ CLS@D -> NxD
				cls_w = _pred @ _W
			else:
				# get Top-5 classes and select their CAMs
				_cls = _classes[:, :5]
				cls_w = _W[_cls]

			cams = (cpu_convs[:, None] * cls_w[..., None, None]).mean(axis=1).sum(axis=1, keepdims=True)
		###### end CAM #####

		for X, conv, att, fin_conv, cam, cls in zip(self._cur_X, convs, atts, convs+atts, cams, _classes):
			fig, axs = plt.subplots(2,2, figsize=(16,9), squeeze=False)

			fig.suptitle(f"Predicted Top-5 Class-IDs: {', '.join(map(str, cls[:5]))}")
			_x = _prepare_back(X)
			axs[0, 0].imshow(_x)
			axs[0, 0].set_title("Input")
			arrs = [
				(_threshold(_normalize(conv)), "conv"),
				# ((_threshold(_normalize(conv)) + _threshold(_normalize(cam))) / 2, "conv+cam"),
				(_threshold((np.sqrt(_normalize(conv)) * _normalize(cam))), "conv*cam"),
				# (_normalize(att), "attention"),
				# (_normalize(fin_conv), "result"),
				(_threshold(_normalize(cam)), "cam"),
			]

			size = _x.shape[:-1]
			for i, (arr, title) in enumerate(arrs, 1):
				ax = axs[np.unravel_index(i, axs.shape)]
				ax.set_title(title)
				# size = None
				if size is not None:
					arr = resize(arr, size, mode="edge")

				ax.imshow(_x, alpha=1.0)
				ax.imshow(arr, vmin=0, vmax=1, alpha=0.8)

			plt.show()
			plt.close()


def _threshold(arr):
	mask = (arr >= arr.mean()).astype(arr.dtype)
	return arr * mask
