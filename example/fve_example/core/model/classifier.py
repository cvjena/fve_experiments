import abc
import chainer
import chainer.functions as F
import chainer.links as L
import logging
import numpy as np

from chainer.serializers import load_npz
from chainer_addons.functions import smoothed_cross_entropy
from cvmodelz import classifiers
from functools import wraps
from functools import partial
from os.path import isfile

from fve_layer.backends.chainer import links as fve_links
from fve_example import utils

def _unpack(var):
	return var[0] if isinstance(var, tuple) else var


def tuple_return(method):

	@wraps(method)
	def inner(self, *args, **kwargs):
		res = method(self, *args, **kwargs)
		if not isinstance(res, tuple):
			res = res,
		return res

	return inner


class BaseFVEClassifier(abc.ABC):
	FVE_CLASSES = dict(em=fve_links.FVELayer, grad=fve_links.FVELayer_noEM)


	@classmethod
	def kwargs(cls, opts) -> dict:
		return dict(
			fve_type=opts.fve_type,
			comp_size=opts.comp_size,
			post_fve_size=opts.post_fve_size,
			n_components=opts.n_components,

			init_mu=opts.init_mu,
			init_sig=opts.init_sig,
			only_mu_part=opts.only_mu_part,
			no_gmm_update=opts.no_gmm_update,

			ema_alpha=opts.ema_alpha,
			aux_lambda=opts.aux_lambda,
			normalize=opts.normalize,
			mask_features=opts.mask_features,
		)


	def __init__(self, *args,
		fve_type: str, comp_size: int, n_components: int,
		post_fve_size: int = -1,
		init_mu=None, init_sig=None,
		only_mu_part=False, no_gmm_update=False,
		ema_alpha=0, aux_lambda=0,
		normalize=False, mask_features=True,
		**kwargs):
		super().__init__(*args, **kwargs)

		self.fve_type = fve_type
		self.n_components = n_components
		self.init_mu = init_mu
		self.init_sig = init_sig
		self.only_mu_part = only_mu_part
		self.no_gmm_update = no_gmm_update
		self.ema_alpha = ema_alpha

		self._output_size = self.model.meta.feature_size

		with self.init_scope():
			self.add_persistent("mask_features", mask_features)
			self.add_persistent("aux_lambda", aux_lambda)

			self.init_encoding(comp_size, post_fve_size, normalize=normalize)

	def load_model(self, *args, **kwargs):
		kwargs["feat_size"] = kwargs.get("feat_size", self.output_size)
		super().load_model(*args, **kwargs)

		""" we need to do this after loading, since the classifier is
			re-initialized after loading and the aux clf uses the
			classifier's parameters
		"""
		with self.init_scope():
			self.init_aux_clf()

	@property
	def output_size(self):
		return self._output_size

	@property
	def encoding_size(self):
		if self.fve_layer is None:
			return None

		# 2 x K x D
		# or K x D (if only_mu_part is True)
		factor = 1 if self.only_mu_part else 2
		return factor * self.fve_layer.in_size * self.fve_layer.n_components


	def init_aux_clf(self):
		if self.aux_lambda <= 0 or self.fve_layer is None:
			self.aux_clf = None
			return

		self.aux_clf = L.Linear(self.fve_layer.in_size, self.n_classes)


	def _init_pre_fve(self, comp_size) -> int:
		assert not hasattr(self, "pre_fve"), \
			"pre-FVE layer was already initialized!"

		if comp_size < 1:
			self.pre_fve = F.identity
			return self.model.meta.feature_size

		self.pre_fve = L.Convolution2D(
			in_channels=self.model.meta.feature_size,
			out_channels=comp_size,
			ksize=1)

		return comp_size

	def _init_fve_layer(self, in_size):

		assert self.fve_type in self.FVE_CLASSES, \
			f"Unknown FVE type: {self.fve_type}!"

		fve_class = self.FVE_CLASSES[self.fve_type]
		logging.info(f"=== Using {fve_class.__name__} ({self.fve_type}) FVE-layer ===")

		kwargs = dict(
			in_size=in_size,
			n_components=self.n_components,

			init_mu=self.init_mu,
			init_sig=self.init_sig,
		)

		if self.fve_type == "em":
			kwargs["alpha"] = self.ema_alpha

		self.fve_layer = fve_class(**kwargs)

	def _init_post_fve(self, post_fve_size, *, activation=F.relu):
		"""
			Initializes a linear layer to apply it after the
			FVE together with a non-linearity (Relu).

			if post_fve_size > 0: take this as output size for the layer.
			if post_fve_size < 0: take self._encoding_size as output size for the layer.
			if post_fve_size == 0: do not initialize a post-FVE layer.

		"""
		enc_size = self.encoding_size

		if post_fve_size > 0:
			sequence = [L.Linear(in_size=enc_size, out_size=post_fve_size), activation]
			self.post_fve = chainer.Sequential(*sequence)
			self._output_size = post_fve_size

		elif post_fve_size < 0:
			sequence = [L.Linear(in_size=enc_size, out_size=enc_size), activation]
			self.post_fve = chainer.Sequential(*sequence)
			self._output_size = enc_size

		else:
			self.post_fve = F.identity
			self._output_size = enc_size

	def init_encoding(self, comp_size, post_fve_size, *, normalize=False):

		if self.fve_type == "no":
			logging.info("=== FVE is disabled! ===")
			self.fve_layer = self.pre_fve = self.post_fve = None
			return

		fve_insize = self._init_pre_fve(comp_size)
		self._init_fve_layer(fve_insize)
		self.normalize = F.normalize if normalize else F.identity
		self._init_post_fve(post_fve_size)

		logging.info("=== Feature masking is {}abled! ===".format("en" if self.mask_features else "dis"))
		logging.info(f"Encoding size: {self.encoding_size}")
		logging.info(f"Final pre-classification size: {self.output_size}")

	def _transform_feats(self, feats):
		assert feats.ndim == 5, \
			f"Malformed input: {feats.shape=}"

		n, t, c, h, w = feats.shape

		# N x T x C x H x W -> N x T x H x W x C
		feats = F.transpose(feats, (0, 1, 3, 4, 2))
		# N x T x H x W x C -> N x T*H*W x C
		feats = F.reshape(feats, (n, t*h*w, c))

		return feats

	def _reduce_feats(self, feats):

		assert feats.ndim in [2, 3], \
			f"Malformed encoding input for non-FVE: {feats.shape=}"

		if feats.ndim == 2:
			# nothing todo: N x D -> N x D
			return feats

		elif feats.ndim == 3:
			# mean over the T-dimension: N x T x D -> N x D
			return F.mean(feats, axis=1)


	def _report_logL(self, feats):
		return
		# TODO: if need it, it is here
		dist = self.fve_layer.mahalanobis_dist(feats)
		mean_min_dist = F.mean(F.min(dist, axis=-1))

		# avarage over all local features
		logL, _ = self.fve_layer.log_proba(feats, weighted=True)
		avg_logL = F.logsumexp(logL) - self.xp.log(logL.size)

		self.report(logL=avg_logL, dist=mean_min_dist)

	def _get_features(self, X, model, use_pre_fve):
		# Input should be (N, C, H, W)
		assert X.ndim == 4, f"Malformed input: {X.shape=}"

		# returns conv map with shape (N, c, h, w)
		conv_map = model(X, model.meta.conv_map_layer)

		# model.pool      returns (N, c)
		# self.pre_fve    returns (N, c', h, w)

		if not use_pre_fve or self.pre_fve is None:
			pool_func = model.pool
		else:
			pool_func = self.pre_fve

		return pool_func(conv_map)

	def predict_aux(self, feats):
		assert feats.ndim in [2, 4], f"Malformed input: {feats.shape=}"

		if feats.ndim == 4:
			# (N, C, H, W) -> (N, C)
			feats = self.model.pool(feats)

		return self.aux_clf(feats)

	@tuple_return
	def encode(self, feats):

		if self.fve_layer is None:
			return self._reduce_feats(feats),

		assert feats.ndim in [4, 5], \
			f"Malformed encoding input for FVE: {feats.shape=}"

		if feats.ndim == 4:
			# nothing todo: N x C x H x W -> N x T x C x H x W
			feats = F.expand_dims(feats, axis=1)

		feats = self._transform_feats(feats)

		if self.no_gmm_update:
			with chainer.using_config("train", False):
				encoding = self.fve_layer(feats, use_mask=self.mask_features)
		else:
			encoding = self.fve_layer(feats, use_mask=self.mask_features)

		self._report_logL(feats)

		encoding = self.normalize(encoding[:, :self.encoding_size])
		return self.post_fve(encoding),

	@tuple_return
	def extract(self, X, model=None, use_pre_fve=True):
		model = model or self.model
		if self._only_head:
			with utils.eval_mode():
				return self._get_features(X, model, use_pre_fve)
		else:
			return self._get_features(X, model, use_pre_fve)

	def forward(self, *inputs) -> chainer.Variable:
		assert len(inputs) in [2, 3], \
			("Expected 2 (image and label) or"
			"3 (image, parts, and label) inputs, "
			f"got {len(inputs)}!")

		*X, y = inputs

		feats: tuple = self.extract(*X)

		logits: tuple = self.encode(*feats)
		preds: tuple = self.predict(*logits)

		self.report(**self.evaluations(*preds, y=y))

		if self.aux_clf is not None:
			aux_pred = self.predict_aux(*feats)
			self.report(aux_p_accu=F.accuracy(aux_pred, y))
			preds += (aux_pred,)

		loss = self.loss(*preds, y=y)
		self.report(loss=loss)

		# from chainer.computational_graph import build_computational_graph as bg
		# from graphviz import Source

		# g = bg([loss])
		# # with open("loss.dot", "w") as f:
		# # 	f.write(g.dump())
		# s = Source(g.dump())
		# s.render("/tmp/foo.dot", cleanup=True, view=True)
		# import pdb; pdb.set_trace()

		return loss



class GlobalClassifier(BaseFVEClassifier, classifiers.Classifier):

	def loss(self, preds, aux_preds=None, *, y) -> chainer.Variable:

		_loss = partial(self.model.loss, gt=y, loss_func=self.loss_func)
		loss = _loss(preds)

		if aux_preds is None:
			return loss

		aux_loss = _loss(aux_preds)
		self.report(aux_loss=aux_loss)
		return self.aux_lambda * aux_loss + (1 - self.aux_lambda) * loss


	def evaluations(self, preds, *, y) -> dict:
		return dict(accu=F.accuracy(preds, y))

	@tuple_return
	def predict(self, logit):
		return self.model.clf_layer(logit)

class PartsClassifier(BaseFVEClassifier, classifiers.SeparateModelClassifier):

	def load_model(self, *args, finetune: bool = False, **kwargs):
		super().load_model(*args, finetune=finetune, **kwargs)

		if finetune:
			self.model.reinitialize_clf(self.n_classes, self.model.meta.feature_size)

	def evaluations(self, global_preds, part_preds, *, y) -> dict:
		global_accu = self.model.accuracy(global_preds, y)
		part_accu = self.separate_model.accuracy(part_preds, y)

		mean_pred = F.log_softmax(global_preds) + F.log_softmax(part_preds)
		accuracy = F.accuracy(mean_pred, y)

		return dict(accu=accuracy, g_accu=global_accu, p_accu=part_accu)

	def loss(self, global_preds, part_preds, aux_preds=None, *, y) -> chainer.Variable:
		_g_loss = partial(self.model.loss, gt=y, loss_func=self.loss_func)
		_p_loss = partial(self.separate_model.loss, gt=y, loss_func=self.loss_func)

		g_loss = _g_loss(global_preds)
		p_loss = _p_loss(part_preds)

		self.report(g_loss=g_loss, p_loss=p_loss)

		if aux_preds is not None:
			aux_loss = _p_loss(aux_preds)
			self.report(aux_loss=aux_loss)
			p_loss = self.aux_lambda * aux_loss + (1 - self.aux_lambda) * p_loss

		return (g_loss + p_loss) / 2

	@tuple_return
	def predict(self, global_logits, part_logits):
		global_pred = self.model.clf_layer(global_logits)
		part_pred = self.separate_model.clf_layer(part_logits)
		return global_pred, part_pred


	def predict_aux(self, glob_feats, part_feats):
		assert part_feats.ndim in [3, 5], f"Malformed input: {part_feats.shape=}"

		if part_feats.ndim == 5:
			n, t, c, h, w = part_feats.shape

			# (N, T, C, H, W) -> (N*T, C, H, W)
			part_feats = part_feats.reshape(n*t, c, h, w)

			# (N*T, C, H, W) -> (N*T, C)
			part_feats = self.model.pool(part_feats)

			# (N*T, C) -> (N, T, C)
			part_feats = part_feats.reshape(n, t, c)

		# (N, T, C) -> (N, C)
		part_feats = F.mean(part_feats, axis=1)
		return self.aux_clf(part_feats)

	@tuple_return
	def encode(self, glob_feats, part_feats):

		glob_enc = self._reduce_feats(glob_feats)

		if self.fve_layer is None:
			part_enc = self._reduce_feats(part_feats)
		else:
			part_enc, = super().encode(part_feats)

		return glob_enc, part_enc

	@tuple_return
	def extract(self, X, parts):
		glob_feats, = super().extract(X, use_pre_fve=False)

		part_feats = []
		for part in parts.transpose(1,0,2,3,4):
			part_feat, = super().extract(part, self.separate_model)
			part_feats.append(part_feat)

		return glob_feats, F.stack(part_feats, axis=1)
