import chainer
import chainer.functions as F
import chainer.links as L

from cvmodelz import classifiers
from functools import partial

from fve_fgvc import utils
from fve_fgvc.core.model.classifier.base import BaseFVEClassifier

from cluster_parts.core import BoundingBoxPartExtractor
from cluster_parts.core import Corrector
from cluster_parts.utils import ThresholdType
from cluster_parts.utils import FeatureType
from cluster_parts.utils import ClusterInitType
from cluster_parts.utils import operations

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
		self._visualize(convs)
		return convs
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

	def _visualize(self, convs):

		_to_cpu = lambda arr: chainer.cuda.to_cpu(chainer.as_array(arr))
		extractor = get_extractor()

		def _normalize(arr, chan_axis=2, channel_wise=False):
			arr = _to_cpu(arr)
			return operations.normalize(arr, axis=chan_axis, channel_wise=channel_wise)

		def _prepare_back(x):
			x = _to_cpu(x)
			x = (x + 1) / 2
			return x.transpose(1,2,0)

		cpu_convs = _to_cpu(convs)
		SOFT_ASSIGNMENT = True

		###### CAM #####
		with chainer.using_config("train", False), chainer.no_backprop_mode():
			_feats = super().encode(convs)
			_preds = self.predict(*_feats)
			_pred = _to_cpu(F.softmax(_preds[0]))
			_W = _to_cpu(self.model.clf_layer.W)
			_classes = np.argsort(-_pred, axis=-1)

			if SOFT_ASSIGNMENT:
				# NxCLS @ CLS@D -> NxD
				cls_w = _pred @ _W
				# NxD -> Nx1xD
				cls_w = np.expand_dims(cls_w, axis=1)
			else:
				# get Top-5 classes and select their CAMs
				_cls = _classes[:, :5]
				# Nx5xD
				cls_w = _W[_cls]

			# Nx1xDxHxW * Nx(1 or 5)xDx1x1 -> Nx(1 or 5)xDxHxW
			weighted_conv = cpu_convs[:, None] * cls_w[..., None, None]
			# Nx(1 or 5)xDxHxW -[mean]-> NxDxHxW -[sum]-> Nx1xHxW
			cams = weighted_conv.mean(axis=1)#.sum(axis=1, keepdims=True)
		###### end CAM #####

		for X, conv, cam, cls in zip(self._cur_X, cpu_convs, cams, _classes):
			fig, axs = plt.subplots(2,2, figsize=(16,9), squeeze=False)

			fig.suptitle(f"Predicted Top-5 Class-IDs: {', '.join(map(str, cls[:5]))}")
			_x = _prepare_back(X)
			axs[0, 0].imshow(_x)
			axs[0, 0].set_title("Input")
			arrs = [
				(conv, "conv"),
				(None, "conv+cam"),
				# ((conv + cam) / 2, "conv+cam"),
				# ((np.sqrt(conv * cam)), "conv*cam"),
				# (att, "attention"),
				# (fin_conv, "result"),
				(cam, "cam"),
			]

			size = _x.shape[:-1]
			for i, (arr, title) in enumerate(arrs, 1):
				ax = axs[np.unravel_index(i, axs.shape)]
				if arr is None:
					ax.axis("off")
					continue
				# CxHxW -> HxWxC
				arr = arr.transpose(1,2,0)
				l2_arr = operations.l2_norm(arr)
				norm_l2_arr = _normalize(l2_arr)

				# size = None
				if size is not None:
					norm_l2_arr = resize(norm_l2_arr, size, mode="edge", preserve_range=True)

				# bring the values to the range (0, 1)
				# norm_l2_arr = _normalize(norm_l2_arr)

				parts = extractor(_x, norm_l2_arr)

				norm_l2_arr = extractor.corrector(norm_l2_arr)
				centers, labs = extractor.cluster_saliency(_x, norm_l2_arr)
				thresh_mask = extractor.thresh_type(_x, norm_l2_arr)
				# parts = extractor.get_boxes(centers, labs, norm_l2_arr)



				if isinstance(thresh_mask, (int, float, norm_l2_arr.dtype.type)):
					thresh_mask = l2_arr > thresh_mask

				norm_l2_arr[~thresh_mask] = 0
				ax.set_title(f"{title} [{(thresh_mask).sum():,d} pixels ({thresh_mask.mean():.2%}) selected]")

				ax.imshow(_x, alpha=0.8)
				ax.imshow(norm_l2_arr, alpha=0.8)
				ax.imshow(labs, alpha=0.4)

				for part_id, ((x,y), w, h) in parts:
					ax.add_patch(plt.Rectangle((x,y), w, h, fill=False))

			plt.show()
			plt.close()



def get_extractor(
	gamma=0.7,
	sigma=5.0,
	K=4,
	fit_object=False,
	thresh_type=ThresholdType.MEAN,
	# thresh_type=ThresholdType.PRECLUSTER,
	comp=[
		FeatureType.COORDS,
		FeatureType.SALIENCY,
		# FeatureType.RGB,
	],
	):

	return BoundingBoxPartExtractor(
		corrector=Corrector(gamma=gamma, sigma=sigma),

		K=K,
		optimal=True,
		fit_object=fit_object,

		thresh_type=thresh_type,
		cluster_init=ClusterInitType.MAXIMAS,
		feature_composition=comp,
	)


def _threshold(arr):
	mask = (arr >= arr.mean()).astype(arr.dtype)
	return arr * mask
