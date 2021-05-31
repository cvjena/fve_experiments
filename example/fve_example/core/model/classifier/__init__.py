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


class BaseFVEClassifier(abc.ABC):
	FVE_CLASSES = dict(em=fve_links.FVELayer, grad=fve_links.FVELayer_noEM)
	N_PARTS = dict(
		L1_full=4,
		L1_pred=4,
		GT2=4,
		GT=15,

		UNI2x2=2**2,
		UNI3x3=3**2,
		UNI4x4=4**2,
		UNI5x5=5**2,
	)

	@classmethod
	def kwargs(cls, opts) -> dict:

		return dict(
			feature_aggregation=opts.feature_aggregation,
			n_parts=cls.N_PARTS.get(opts.parts, 1),

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
		feature_aggregation="mean", n_parts=1,
		**kwargs):
		super().__init__(*args, **kwargs)

		self.feature_aggregation = feature_aggregation
		self.fve_type = fve_type
		self.n_components = n_components
		self.init_mu = init_mu
		self.init_sig = init_sig
		self.only_mu_part = only_mu_part
		self.no_gmm_update = no_gmm_update
		self.ema_alpha = ema_alpha

		feat_size = self.model.meta.feature_size
		if self.fve_type == "no" and self.feature_aggregation == "concat":
			self._output_size = feat_size * n_parts
		else:
			self._output_size = feat_size

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

		# input size will be set at the first iteration
		# layer initialization will be done at runtime as well
		self.aux_clf = L.Linear(self.n_classes)


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
		""" Transforms a 5D conv map (NxTxCxHxW) to 3D local features (NxT*H*WxC) """

		assert feats.ndim == 5, \
			f"Malformed input: {feats.shape=}"

		n, t, c, h, w = feats.shape

		# N x T x C x H x W -> N x T x H x W x C
		feats = F.transpose(feats, (0, 1, 3, 4, 2))
		# N x T x H x W x C -> N x T*H*W x C
		feats = F.reshape(feats, (n, t*h*w, c))

		return feats

	def _report_logL(self, feats):
		return
		# TODO: if need it, it is here
		dist = self.fve_layer.mahalanobis_dist(feats)
		mean_min_dist = F.mean(F.min(dist, axis=-1))

		# avarage over all local features
		logL, _ = self.fve_layer.log_proba(feats, weighted=True)
		avg_logL = F.logsumexp(logL) - self.xp.log(logL.size)

		self.report(logL=avg_logL, dist=mean_min_dist)

	def _get_features(self, X, model):
		# Input should be (N, 3, H, W)
		assert X.ndim == 4, f"Malformed input: {X.shape=}"

		# returns conv map with shape (N, C, h, w)
		return model(X, model.meta.conv_map_layer)

	def call_pre_fve(self, convs):
		assert convs.ndim in [4,5], \
			f"Malformed pre-FVE input: {convs.shape=}"
		assert self.pre_fve is not None, \
			"pre-FVE was not initialized!"

		if convs.ndim == 4:
			return self.pre_fve(convs)

		n, t, c, h, w = convs.shape
		convs = convs.reshape(n*t, c, h, w)
		return self.pre_fve(convs).reshape(n, t, -1, h, w)


	def fve_encode(self, convs):
		""" Encodes 5D conv map (NxTxCxHxW) with the FVELayer """

		assert convs.ndim == 5, \
			f"Malformed FVE encoding input: {convs.shape=}"
		assert self.fve_layer is not None

		convs = self.call_pre_fve(convs)
		convs = self._transform_feats(convs)

		if self.no_gmm_update:
			with chainer.using_config("train", False):
				encoding = self.fve_layer(convs, use_mask=self.mask_features)
		else:
			encoding = self.fve_layer(convs, use_mask=self.mask_features)

		self._report_logL(convs)

		encoding = self.normalize(encoding[:, :self.encoding_size])
		return self.post_fve(encoding)

	@utils.tuple_return
	def extract(self, X, model=None):
		""" extracts from a batch of images (Nx3xHxW) a batch of conv maps (NxCxhxw) """

		model = model or self.model
		if self._only_head:
			with utils.eval_mode():
				return self._get_features(X, model)
		else:
			return self._get_features(X, model)

	def forward(self, *inputs) -> chainer.Variable:
		assert len(inputs) in [2, 3], \
			("Expected 2 (image and label) or"
			"3 (image, parts, and label) inputs, "
			f"got {len(inputs)}!")

		*X, y = inputs

		convs: tuple = self.extract(*X)

		feats: tuple = self.encode(*convs)
		preds: tuple = self.predict(*feats)

		if self.aux_clf is not None:
			aux_pred = self.predict_aux(*convs)
			preds += (aux_pred,)

		self.report(**self.evaluations(*preds, y=y))
		loss = self.loss(*preds, y=y)
		self.report(loss=loss)

		# from chainer.computational_graph import build_computational_graph as bg
		# from graphviz import Source

		# g = bg([loss])
		# # with open("loss.dot", "w") as f:
		# # 	f.write(g.dump())
		# s = Source(g.dump())
		# s.render("/tmp/foo.dot", cleanup=True, view=True)
		# exit(1)

		return loss

	@abc.abstractmethod
	def loss(self, *args, **kwargs):
		pass

	@abc.abstractmethod
	def evaluations(self, *args, **kwargs):
		pass

	@abc.abstractmethod
	def predict(self, *args, **kwargs):
		pass

	@abc.abstractmethod
	def encode(self, *args, **kwargs):
		pass

	@abc.abstractmethod
	def predict_aux(self, *args, **kwargs):
		pass


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

class PartsClassifier(BaseFVEClassifier, classifiers.SeparateModelClassifier):

	@classmethod
	def kwargs(cls, opts) -> dict:
		kwargs = super().kwargs(opts)
		kwargs["copy_mode"] = opts.copy_mode
		return kwargs

	def load_model(self, *args, finetune: bool = False, **kwargs):
		super().load_model(*args, finetune=finetune, **kwargs)

		if finetune:
			if self.copy_mode == "share":
				clf_name = self.model.clf_layer_name
				new_clf = L.Linear(self.model.meta.feature_size, self.n_classes)
				setattr(self.model, clf_name, new_clf)

			self.model.reinitialize_clf(self.n_classes, self.model.meta.feature_size)

		with self.init_scope():
			self.comb_clf = L.Linear(self.n_classes)

	def loss(self, global_preds, part_preds, combined_pred, aux_preds=None, *, y) -> chainer.Variable:
		_g_loss = partial(self.model.loss, gt=y, loss_func=self.loss_func)
		_p_loss = partial(self.separate_model.loss, gt=y, loss_func=self.loss_func)

		g_loss = _g_loss(global_preds)
		p_loss = _p_loss(part_preds)

		if aux_preds is not None:

			#### This one was used previously,
			#### but does not make sence mathematically
			"""
			p_preds = self.aux_lambda * aux_preds + (1 - self.aux_lambda) * part_preds
			"""
			aux_loss = _p_loss(aux_preds)
			self.report(aux_loss=aux_loss)
			p_loss = self.aux_lambda * aux_loss + (1 - self.aux_lambda) * p_loss

		self.report(g_loss=g_loss, p_loss=p_loss)

		# return (g_loss + p_loss) * 0.5

		#### This one was used previously,
		#### but does not make sence mathematically
		# 2048 N*2048 -> (N+1)*2048
		# 2048 2*K*D -> 2048 + 2*K*D

		comb_loss = _g_loss(combined_pred)
		return ((g_loss + p_loss) * 0.5 + comb_loss) * 0.5

	def evaluations(self, global_preds, part_preds, combined_pred, aux_preds=None, *, y) -> dict:

		global_accu = self.model.accuracy(global_preds, y)
		part_accu = self.separate_model.accuracy(part_preds, y)

		evals = {}
		if aux_preds is not None:
			evals["aux_p_accu"] = F.accuracy(aux_preds, y)

			#### This one was used previously,
			#### but does not make sence mathematically
			# part_preds = self.aux_lambda * aux_preds + (1 - self.aux_lambda) * part_preds

		# combined_pred = F.log_softmax(global_preds) + F.log_softmax(part_preds)
		accu = F.accuracy(combined_pred, y)

		return dict(evals, accu=accu, g_accu=global_accu, p_accu=part_accu)


	@utils.tuple_return
	def predict(self, global_feats, part_feats):
		global_pred = self.model.clf_layer(global_feats)
		part_pred = self.separate_model.clf_layer(part_feats)

		if getattr(self, comb_clf, None) is not None:
			comb_feat = F.concat([global_feats, part_feats], axis=1)
			combined_pred = self.comb_clf(comb_feat)

		else:
			combined_pred = global_pred + part_pred

		return global_pred, part_pred, combined_pred


	def predict_aux(self, glob_convs, part_convs):
		assert part_convs.ndim == 5, \
			f"Malformed aux input: {part_convs.shape=}"

		part_feats = self.pool_encode(part_convs)
		return self.aux_clf(part_feats)


	def pool_encode(self, part_convs):

		n,t,c,h,w = part_convs.shape
		# N x T x C x H x W -> N*T x C x H x W
		part_convs = part_convs.reshape((n*t, c, h, w))

		# N x T x C x H x W -> N*T x C
		part_feats = self.separate_model.pool(part_convs)

		# N*T x C -> N x T x C
		part_feats = part_feats.reshape(n, t, c)

		# N x T x C -> N x C    if aggregation is "mean"
		# N x T x C -> N x T*C  if aggregation is "concat"
		return self._aggregate_feats(part_feats)


	@utils.tuple_return
	def encode(self, glob_convs, part_convs):
		""" Implements the encoding of a 4D global conv map with model's pooling
			and the encoding of a 5D part conv map either with the separate_model's pooling
			or with the FVELayer
		"""
		assert glob_convs.ndim == 4, \
			f"Malformed global conv encoding input: {glob_convs.shape=}"

		assert part_convs.ndim == 5, \
			f"Malformed part conv encoding input: {glob_convs.shape=}"


		if self.fve_layer is None:
			enc_func = self.pool_encode
		else:
			enc_func = self.fve_encode

		# N x C x H x W -> N x C
		glob_feat = self.model.pool(glob_convs)

		# N x T x C x H x W -> N x C or N x T*C or N x 2*C*K
		part_feats = enc_func(part_convs)

		return glob_feat, part_feats

	@utils.tuple_return
	def extract(self, X, parts):
		glob_feats, = super().extract(X)

		part_feats = []
		for part in parts.transpose(1,0,2,3,4):
			part_feat, = super().extract(part, self.separate_model)
			part_feats.append(part_feat)

		return glob_feats, F.stack(part_feats, axis=1)

	def _aggregate_feats(self, feats):

		assert feats.ndim == 3, \
			f"Malformed encoding input for feature aggregation: {feats.shape=}"

		if self.feature_aggregation == "mean":
			# mean over the T-dimension: N x T x D -> N x D
			return F.mean(feats, axis=1)

		elif self.feature_aggregation == "concat":
			# concat all features together: N x T x D -> N x T*D
			n, t, d = feats.shape
			return F.reshape(feats, (n, t*d))

		else:
			raise ValueError(f"Unknown feature aggregation method: {self.feature_aggregation}")
