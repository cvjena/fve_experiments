import chainer
import chainer.functions as F
import chainer.links as L

from cvmodelz import classifiers
from functools import partial

from fve_fgvc import utils
from fve_fgvc.core.model.classifier.base import BaseFVEClassifier

import numpy as np
from matplotlib import pyplot as plt

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
			self._init_attention()

	def _init_attention(self):
		n_convs = self.model.meta.n_conv_maps

		self.query = L.Convolution2D(n_convs, n_convs, ksize=1)
		self.key = L.Convolution2D(n_convs, n_convs, ksize=1)
		self.value = L.Convolution2D(n_convs, n_convs, ksize=1)

		self.gamma = chainer.Parameter(0.0)
		self.gamma.initialize(None)


	def self_attention(self, convs):
		assert convs.ndim == 4, \
			f"invalid input shape: {convs.shape}"
		N, C, H, W = convs.shape

		query = self.query(convs).reshape(N, C, H*W)
		key = self.key(convs).reshape(N, C, H*W)
		value = self.value(convs).reshape(N, C, H*W)

		# energy.shape = N x H * W x H * W
		energy = F.matmul(query, key, transa=True)

		# attention.shape = N x H * W x H * W
		attention = F.softmax(energy, axis=-1)

		out = F.matmul(value, attention, transb=True)
		out = out.reshape(N, C, H, W)

		self.report(gamma=self.gamma)

		self._visualize(convs, self.gamma*out)

		return self.gamma * out + convs

	def _visualize(self, convs, atts):
		def _normalize(arr, chan_axis=0):
			arr = chainer.cuda.to_cpu(chainer.as_array(arr))

			non_chan_axis = tuple([i for i in range(arr.ndim) if i != chan_axis])
			arr -= arr.min(axis=non_chan_axis, keepdims=True)

			_max_vals = arr.max(axis=non_chan_axis, keepdims=True)

			mask = (_max_vals != 0).squeeze()
			if mask.any():
				arr[mask] /= _max_vals[mask]
				arr = arr.sum(axis=chan_axis) / mask.sum()
			else:
				arr = arr.mean(axis=chan_axis)

			# if arr.max() != 0:
			# 	arr /= arr.max()
			return arr

		def _prepare_back(x):
			x = chainer.cuda.to_cpu(chainer.as_array(x))
			x = (x + 1) / 2
			return x.transpose(1,2,0)

		for X, conv, att, fin_conv in zip(self._cur_X, convs, atts, convs+atts):
			fig, axs = plt.subplots(2,2, figsize=(16,9), squeeze=False)

			arrs = [
				(_prepare_back(X), "Input"),
				(_normalize(conv), "conv"),
				(_normalize(att), "attention"),
				(_normalize(fin_conv), "result"),
			]

			for i, (arr, title) in enumerate(arrs):
				ax = axs[np.unravel_index(i, axs.shape)]
				ax.set_title(title)
				ax.imshow(arr, vmin=0, vmax=1)

			plt.show()
			plt.close()

	def extract(self, X, *args, **kwargs):
		self._cur_X = X
		return super().extract(X, *args, **kwargs)


	@utils.tuple_return
	def encode(self, convs):

		att_convs = self.self_attention(convs)
		return super().encode(att_convs)

