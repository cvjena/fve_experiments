import chainer
import chainer.functions as F
import chainer.links as L
import logging
import numpy as np

from functools import partial
from os.path import isfile

from chainer.serializers import load_npz
from chainer_addons.functions import smoothed_cross_entropy

from fve_layer.backends.chainer import links as fve_links

def _unpack(var):
	return var[0] if isinstance(var, tuple) else var


class Classifier(chainer.Chain):

	@classmethod
	def load(cls, opts, *args, **kwargs):
		assert opts.load is not None and isfile(opts.load), \
			"\"load\" parameter is required and must be a file!"

		cont = np.load(opts.load)

		if any(param.startswith("separate_model") for param in cont.files):
			opts.separate_model = True

		opts.aux_lambda = 0.0
		opts.aux_lambda_rate = 1
		opts.mask_features = True
		opts.from_scratch = True
		opts.label_smoothing = 0.1

		if "fve_layer/w" in cont:
			opts.fve_type = "em" if "fve_layer/t" in cont else "grad"

			fv_insize, n_comps = cont["fve_layer/mu"].shape
			if "pre_fve/W" not in cont:
				# there is no "pre_fve" layer
				fv_insize = -1

			opts.n_components = n_comps
			opts.comp_size = fv_insize

			logging.info(f"=== Loading Classifier with {opts.fve_type}-based FVE-Layer "
				f"({n_comps}x{fv_insize}) ===")

		else:
			opts.fve_type = "no"

			logging.info("=== Loading Classifier without FVE-Layer ===")

		clf = cls(opts, *args, **kwargs)

		logging.info(f"Loading classifier weights from \"{opts.load}\"")
		load_npz(opts.load, clf)

		del clf.model.pool
		clf.model.pool = F.identity

		return clf


	@classmethod
	def new(cls, *args, **kwargs):
		return cls(*args, **kwargs)

	def __init__(self, args, model, n_classes, default_weights):
		super(Classifier, self).__init__()

		with self.init_scope():
			self.model = model
			self.separate_model = self._init_sep_model(args)
			self.init_encoding(args)

			self.add_persistent("aux_lambda", args.aux_lambda)
			self.init_aux_clf(n_classes)

		self._load(args, default_weights, n_classes)
		self._init_loss(args, n_classes)

		self._only_clf = args.only_clf
		self._no_gmm_update = args.no_gmm_update


	def report(self, **values):
		chainer.report(values, self)

	def init_aux_clf(self, n_classes):

		if self.aux_lambda > 0 and self.fve_layer is not None:
			self.aux_clf = L.Linear(self.fve_layer.in_size, n_classes)

		else:
			self.aux_clf = None

	def _init_pre_fve(self, *, comp_size: int) -> int:
		""" Initializes the linear feature size reduction
			(if comp_size > 1).

			Returns the input size for the FVE initialization
		"""
		if comp_size < 1:
			self.pre_fve = F.identity
			return self.model.meta.feature_size

		self.pre_fve = L.Convolution2D(
			in_channels=self.model.meta.feature_size,
			out_channels=comp_size,
			ksize=1)
		return comp_size


	def _init_fve(self, *, fve_type, fv_insize, n_components, init_mu, init_sig, only_mu_part=False, ema_alpha=0.99):

		fve_classes = dict(
			em=fve_links.FVELayer,
			grad=fve_links.FVELayer_noEM
		)
		assert fve_type in fve_classes, \
			f"Unknown FVE type: {fve_type}!"

		fve_class = fve_classes[fve_type]
		logging.info(f"=== Using {fve_class.__name__} ({fve_type}) FVE-layer ===")

		fve_kwargs = dict(
			in_size=fv_insize,
			n_components=n_components,

			init_mu=init_mu,
			init_sig=init_sig,

		)

		if fve_type == "em":
			fve_kwargs["alpha"] = ema_alpha

		self.fve_layer = fve_class(**fve_kwargs)

		factor = 1 if only_mu_part else 2

		# 2 x K x D or 1 x K x D (if only_mu_part is True)
		self._encoding_size = factor * n_components * fv_insize


	def _init_post_fve(self, *, post_fve_size, activation=F.relu):
		"""
			Initializes a linear layer to apply it after the
			FVE together with a non-linearity (Relu).

			if post_fve_size > 0: take this as output size for the layer.
			if post_fve_size < 0: take self._encoding_size as output size for the layer.
			if post_fve_size == 0: do not initialize a post-FVE layer.

		"""
		if post_fve_size > 0:
			self.post_fve = chainer.Sequential(
				L.Linear(in_size=self._encoding_size, out_size=post_fve_size),
				activation
			)
			self._output_size = post_fve_size
		elif post_fve_size < 0:
			self.post_fve = chainer.Sequential(
				L.Linear(in_size=self._encoding_size, out_size=self._encoding_size),
				activation
			)
			self._output_size = self._encoding_size
		else:
			self.post_fve = F.identity
			self._output_size = self._encoding_size


	def init_encoding(self, args):
		""" Initializes the linear feature size reduction
			(if comp_size > 1)
			and the FVE Layer according to the given FVE type.
		"""

		if args.fve_type == "no":
			logging.info("=== FVE is disabled! ===")
			self.fve_layer = self.pre_fve = self.post_fve = None
			self._output_size = self.model.meta.feature_size

		else:
			fv_insize = self._init_pre_fve(
				comp_size=args.comp_size
			)

			self._init_fve(
				fve_type=args.fve_type,
				fv_insize=fv_insize,
				n_components=args.n_components,

				init_mu=args.init_mu,
				init_sig=args.init_sig,
				only_mu_part=args.only_mu_part,

				ema_alpha=args.ema_alpha,
			)

			self.normalize = F.normalize if args.normalize else F.identity

			self._init_post_fve(
				post_fve_size=args.post_fve_size
			)


		logging.info(f"Final pre-classification size: {self.output_size}")
		self.add_persistent("mask_features", args.mask_features)
		logging.info("=== Feature masking is {}abled! ===".format("en" if self.mask_features else "dis"))

	def _init_loss(self, args, n_classes):
		smoothing = args.label_smoothing
		if smoothing > 0:
			assert smoothing < 1, \
				"Label smoothing factor must be less than 1!"
			self.loss = partial(smoothed_cross_entropy,
				N=n_classes,
				eps=smoothing)

		else:
			self.loss = F.softmax_cross_entropy

	def _init_sep_model(self, args):

		if args.parts != "GLOBAL" and args.separate_model:
			logging.info("Created a separate model for global image processing")
			return self.model.copy(mode="copy")

		else:
			logging.warning("No separate model for global image processing was created")
			return None

	def _load(self, args, weights, n_classes):

		load_path = args.load_path or ""
		self._load_weights(args, self.model, weights, n_classes,
			path=load_path + "model/",
			feat_size=self.output_size)

		if self.separate_model is not None:
			for possible_path in ["separate_model", "model"]:
				path = f"{load_path}{possible_path}/"
				try:
					self._load_weights(args, self.separate_model, weights, n_classes,
						path=path,
						feat_size=self.model.meta.feature_size)
					break
				except KeyError:
					logging.error(f"Failed to load weights with path \"{path}\". trying other paths...")
					pass

		if not args.load:
			return

		if isinstance(self.pre_fve, chainer.Link):
			logging.info("Loading weights for preFVE-Layer")
			try:
				load_npz(args.load, self.pre_fve, path="pre_fve/")
			except KeyError:
				logging.error(f"Failed to load pre-FVE layer weights from \"{args.load}\"")

		if isinstance(self.fve_layer, chainer.Link):
			logging.info("Loading weights for FVE-Layer")
			try:
				load_npz(args.load, self.fve_layer, path="fve_layer/")
			except KeyError:
				logging.error(f"Failed to load FVE layer weights from \"{args.load}\"")




	def _load_weights(self, args, model, weights, n_classes,
		path="", feat_size=None):
		feat_size = feat_size or self.output_size

		if args.from_scratch:
			logging.info("No weights loaded, training from scratch!")
			model.reinitialize_clf(
				n_classes=n_classes,
				feat_size=feat_size)
			return

		loader = model.load_for_finetune
		load_path = ""
		msg = f"Loading default pre-trained weights from \"{weights}\""

		if args.load:
			loader = model.load_for_inference
			weights = args.load
			load_path = path
			msg = f"Loading already fine-tuned weights from \"{weights}\""

		elif args.weights:
			weights = args.weights
			msg = f"Loading custom pre-trained weights from \"{weights}\""

		# assert weights is not None
		logging.info(msg + (f" ({load_path})" if load_path else ""))
		loader(
			weights=weights,
			n_classes=n_classes,
			path=load_path,
			feat_size=feat_size,

			strict=args.load_strict,
			headless=args.headless,
		)

	def _transform_feats(self, feats):

		n, t, c, h, w = feats.shape

		# N x T x C x H x W -> N x T x H x W x C
		feats = F.transpose(feats, (0, 1, 3, 4, 2))
		# N x T x H x W x C -> N x T*H*W x C
		feats = F.reshape(feats, (n, t*h*w, c))

		return feats

	def encode(self, feats):

		if feats.ndim == 2:
			# reshape, to match N x T x D with T=1
			# N x D -> N x 1 x D
			feats = F.expand_dims(feats, axis=1)

		if self.fve_layer is None:
			# mean over the T-dimension: N x T x D -> N x D
			return F.mean(feats, axis=1)

		feats = self._transform_feats(feats)

		if self._no_gmm_update:
			with chainer.using_config("train", False):
				encoding = self.fve_layer(feats, use_mask=self.mask_features)
		else:
			encoding = self.fve_layer(feats, use_mask=self.mask_features)

		dist = self.fve_layer.mahalanobis_dist(feats)
		mean_min_dist = F.mean(F.min(dist, axis=-1))

		# avarage over all local features
		logL, _ = self.fve_layer.log_proba(feats, weighted=True)
		avg_logL = F.logsumexp(logL) - self.xp.log(logL.size)

		self.report(
			logL=avg_logL,
			dist=mean_min_dist
		)

		encoding = self.normalize(encoding[:, :self._encoding_size])
		return self.post_fve(encoding)

	def _get_conv_map(self, x, model=None):
		model = model or self.model
		return _unpack(model(x,
					layer_name=model.meta.conv_map_layer))

	def _call_pre_fve(self, convs, model=None):
		model = model or self.model
		if self.pre_fve is None:
			return model.pool(convs)

		return self.pre_fve(convs)

	def get_global_features(self, X):

		conv_map = self._get_conv_map(X, model=self.separate_model)
		if self.separate_model is not None:
			return self.separate_model.pool(conv_map)

		return self._call_pre_fve(conv_map)


	def extract_global(self, X):
		if self._only_clf:
			with chainer.no_backprop_mode(), chainer.using_config("train", False):
				feats = self.get_global_features(X)
		else:
			feats = self.get_global_features(X)

		if self.separate_model is not None:
			# it is already "encoded" with the separate model's pooling
			return feats

		return self.encode(feats)


	def get_part_features(self, parts):
		n, t, c, h, w = parts.shape

		_parts = parts.reshape(n*t, c, h, w)
		part_convs = self._get_conv_map(_parts)


		part_convs = self._call_pre_fve(part_convs)

		_n, *rest = part_convs.shape

		return part_convs.reshape(n, t, *rest)

	def extract_parts(self, parts):
		if parts is None:
			return None

		if self._only_clf:
			with chainer.no_backprop_mode(), chainer.using_config("train", False):
				part_convs = self.get_part_features(parts)
		else:
			part_convs = self.get_part_features(parts)

		# store it for the auxilary classifier
		self.part_convs = part_convs

		enc = self.encode(part_convs)

		return enc

	def extract(self, X, parts=None):
		glob_convs = self.extract_global(X)
		part_convs = self.extract_parts(parts)
		return glob_convs, part_convs

	def predict_global(self, y, global_logit):
		model = self.separate_model or self.model

		pred = model.clf_layer(global_logit)

		self.report(
			g_accu=F.accuracy(pred, y),
			g_loss=self.loss(pred, y)
		)

		return pred


	def predict_parts(self, y, part_logits):
		if part_logits is None:
			return None

		pred = self.model.clf_layer(part_logits)

		self.report(
			p_accu=F.accuracy(pred, y),
			p_loss=self.loss(pred, y)
		)

		if getattr(self, "aux_clf") is None:
			self.part_convs = None
			return pred

		n, t, c, h, w = self.part_convs.shape
		map_feats = F.mean(self.part_convs, axis=(3,4))

		_map_feats = F.reshape(map_feats, shape=(n*t, c))
		_aux_pred = self.aux_clf(_map_feats)
		aux_pred = F.reshape(_aux_pred, shape=(n, t, -1))
		aux_pred = F.sum(aux_pred, axis=1)

		final_pred = pred * (1 - self.aux_lambda) + aux_pred * self.aux_lambda

		self.report(
			aux_p_accu=F.accuracy(final_pred, y),
			aux_p_loss=self.loss(final_pred, y),
			aux_lambda=self.aux_lambda,
		)

		return final_pred

	def predict(self, y, global_logit, part_logits=None):

		glob_pred = self.predict_global(y, global_logit)
		part_pred = self.predict_parts(y, part_logits)

		return glob_pred, part_pred

	def get_loss(self, y, glob_pred, part_pred=None):
		if part_pred is None:
			loss = self.loss(glob_pred, y)
			accu = F.accuracy(glob_pred, y)
			# f1score = F.f1_score(glob_pred, y)[0]

		else:
			pred = part_pred + glob_pred
			loss  = 0.50 * self.loss(pred, y)
			loss += 0.25 * self.loss(glob_pred, y)
			loss += 0.25 * self.loss(part_pred, y)
			accu = F.accuracy(pred, y)
			# f1score = F.f1_score(pred, y)[0]

		# f1score = self.xp.nanmean(f1score.array)
		self.report(
			accu=accu,
			# f1=f1score,
			loss=loss,
		)
		return loss


	def forward(self, *inputs):
		assert len(inputs) in [2, 3], \
			("Expected 2 (image and label) or"
			"3 (image, parts, and label) inputs, "
			f"got {len(inputs)}!")

		*X, y = inputs
		logits = self.extract(*X)

		preds = self.predict(y, *logits)
		loss = self.get_loss(y, *preds)


		# from chainer.computational_graph import build_computational_graph as bg
		# g = bg([loss])

		# with open("loss.dot", "w") as f:
		# 	f.write(g.dump())

		# exit(-1)

		# from graphviz import Source
		# s = Source(g.dump())
		# s.render("/tmp/foo.dot", cleanup=True, view=True)
		# import pdb; pdb.set_trace()

		return loss


	@property
	def output_size(self):
		return self._output_size


class FeatureAugmentClassifier(Classifier):

	def __init__(self, *args, **kwargs):
		super(FeatureAugmentClassifier, self).__init__(*args, **kwargs)

		self.augment_fraction = 0.25


	def _augment_feats(self, feats):

		n, t, c, h, w = feats.shape

		sampled, _ = self.fve_layer.sample(n*t*h*w)

		# N*T*H*W x C -> N x T x H x W x C
		sampled = sampled.reshape(n, t, h, w, c)
		# N x T x H x W x C -> N x T x C x H x W
		sampled = sampled.transpose(0, 1, 4, 2, 3)
		sampled = self.xp.array(sampled)

		n_augment = int(t*h*w * self.augment_fraction)

		mask = self.xp.zeros((n, t, 1, h, w), dtype=np.bool)
		for i in range(n):
			idxs = np.random.choice(t*h*w,
				size=n_augment,
				replace=False)
			ts, hs, ws = np.unravel_index(idxs, (t, h, w))
			mask[i, ts, 0, hs, ws] = 1

		aug_feats = feats * ~mask + sampled * mask
		return aug_feats

	def encode(self, feats):

		if chainer.config.train and self.augment_fraction > 0:
			feats = self._augment_feats(feats)

		return super(FeatureAugmentClassifier, self).encode(feats)
