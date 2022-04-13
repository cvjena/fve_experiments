import chainer
import chainer.functions as F
import chainer.links as L

from cvmodelz import classifiers
from functools import partial

from fve_fgvc import utils
from fve_fgvc.core.model.classifier import base


class GlobalClassifier(base.BaseFVEClassifier,
					   classifiers.Classifier):

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

		return self.gamma * out + convs


	@utils.tuple_return
	def encode(self, convs):
		if self.within_encode_scope():
			return super().encode(convs)

		with self.encode_scope():
			#convs = self.self_attention(convs)
			return super().encode(convs)
