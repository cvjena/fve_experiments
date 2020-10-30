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
			self.init_encoding(args)

			self.add_persistent("aux_lambda", args.aux_lambda)
			self.init_aux_clf(n_classes)

		self._load_weights(args, default_weights, n_classes)

		self._init_sep_model(args, n_classes)
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

	def init_encoding(self, args):
		""" Initializes the linear feature size reduction
			(if comp_size > 1)
			and the FVE Layer according to the given FVE type.
		"""

		if args.fve_type == "no":
			logging.info("=== FVE is disabled! ===")
			self.fve_layer = self.pre_fve = None
			self._output_size = self.model.meta.feature_size

		else:
			n_comps = args.n_components

			if args.comp_size < 1:
				self.pre_fve = F.identity
				fv_insize = self.model.meta.feature_size

			else:
				self.pre_fve = L.Convolution2D(
					in_channels=self.model.meta.feature_size,
					out_channels=args.comp_size,
					ksize=1)
				fv_insize = args.comp_size


			fve_classes = dict(
				em=fve_links.FVELayer,
				grad=fve_links.FVELayer_noEM
			)
			assert args.fve_type in fve_classes, \
				f"Unknown FVE type: {args.fve_type}!"

			fve_class = fve_classes[args.fve_type]
			logging.info(f"=== Using {fve_class.__name__} ({args.fve_type}) FVE-layer ===")

			fve_kwargs = dict(
				in_size=args.comp_size,
				n_components=args.n_components
			)

			if args.fve_type == "em":
				fve_kwargs["alpha"] = args.ema_alpha

			self.fve_layer = fve_class(**fve_kwargs)

			self._output_size = 2 * fv_insize * n_comps

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

	def _init_sep_model(self, args, n_classes):

		sep_model = None
		if args.parts != "GLOBAL" and args.separate_model:
			logging.info("Created a separate model for global image processing")
			sep_model = self.model.copy(mode="copy")
			sep_model.reinitialize_clf(n_classes, self.model.meta.feature_size)
		else:
			logging.warning("No separate model for global image processing was created")

		with self.init_scope():
			self.separate_model = sep_model

	def _load_weights(self, args, weights, n_classes):

		if args.from_scratch:
			logging.info("No weights loaded, training from scratch!")
			self.model.reinitialize_clf(
				n_classes=n_classes,
				feat_size=self.output_size)
			return

		loader = self.model.load_for_finetune
		msg = f"Loading default pre-trained weights from \"{weights}\""

		if args.load:
			loader = self.model.load_for_inference
			weights = args.load
			msg = f"Loading already fine-tuned weights from \"{weights}\""

		elif args.weights:
			weights = args.weights
			msg = f"Loading custom pre-trained weights from \"{weights}\""

		# assert weights is not None
		logging.info(msg)
		loader(
			weights=weights,
			n_classes=n_classes,
			path=args.load_path or "",
			feat_size=self.output_size,

			strict=args.load_strict,
			headless=args.headless,
		)

	def _mse_gmm_params(self, feats):
		if self.fve_layer.n_components != 1:
			return

		mu, sig = self.fve_layer.mu[:, 0], self.fve_layer.sig[:, 0]
		mu, sig = getattr(mu, "array", mu), getattr(sig, "array", sig)

		mask = self.fve_layer.get_mask(feats, use_mask=self.mask_features)

		selected = feats.array[mask].reshape(-1, feats.shape[-1])

		feat_mean = selected.mean(axis=0)
		feat_std = selected.std(axis=0)

		xp = self.xp

		self.report(
			mse_mu=xp.mean((mu - feat_mean)**2),
			mse_sig=xp.mean((xp.sqrt(sig) - feat_std)**2),
		)

	def encode(self, feats):

		if self.fve_layer is None:
			return F.mean(feats, axis=1)

		n, t, c, h, w = feats.shape

		# N x T x C x H x W -> N x T x H x W x C
		feats = F.transpose(feats, (0, 1, 3, 4, 2))
		# N x T x H x W x C -> N x T*H*W x C
		feats = F.reshape(feats, (n, t*h*w, c))

		if self._no_gmm_update:
			with chainer.using_config("train", False):
				logits = self.fve_layer(feats, use_mask=self.mask_features)
		else:
			logits = self.fve_layer(feats, use_mask=self.mask_features)

		self._mse_gmm_params(feats)

		return logits

	def _get_conv_map(self, x, model=None):
		model = model or self.model
		return _unpack(model(x,
					layer_name=model.meta.conv_map_layer))

	def extract_global(self, X):
		conv_map = self._get_conv_map(X, model=self.separate_model)

		if self.separate_model is not None:
			return self.separate_model.pool(conv_map)

		if self.fve_layer is None:
			return self.model.pool(conv_map)

		feats = self.pre_fve(conv_map)
		feats = F.expand_dims(feats, axis=1)

		return self.encode(feats)

	def get_part_features(self, parts):
		part_convs = []
		_pre_fve = self.model.pool if self.pre_fve is None else self.pre_fve
		n, t, c, h, w = parts.shape

		_parts = parts.reshape(n*t, c, h, w)
		part_convs = self._get_conv_map(_parts)
		part_convs = _pre_fve(part_convs)

		_n, *rest = part_convs.shape

		return part_convs.reshape(n, t, *rest)

	def extract_parts(self, parts):
		if parts is None:
			return None

		# store it for the auxilary classifier
		self.part_convs = part_convs = self.get_part_features(parts)

		return self.encode(part_convs)

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

		else:
			pred = part_pred + glob_pred
			loss  = 0.50 * self.loss(pred, y)
			loss += 0.25 * self.loss(glob_pred, y)
			loss += 0.25 * self.loss(part_pred, y)
			accu = F.accuracy(pred, y)


		self.report(accu=accu, loss=loss)
		return loss


	def forward(self, *inputs):
		assert len(inputs) in [2, 3], \
			("Expected 2 (image and label) or"
			"3 (image, parts, and label) inputs, "
			f"got {len(inputs)}!")

		*X, y = inputs

		if self._only_clf:
			with chainer.no_backprop_mode(), chainer.using_config("train", False):
				logits = self.extract(*X)
		else:
			logits = self.extract(*X)

		preds = self.predict(y, *logits)
		loss = self.get_loss(y, *preds)

		# from chainer.computational_graph import build_computational_graph as bg
		# from graphviz import Source
		# g = bg([loss])

		# with open("")
		# s = Source(g.dump())
		# s.render("/tmp/foo.dot", cleanup=True, view=True)
		# import pdb; pdb.set_trace()

		return loss




	@property
	def output_size(self):
		return self._output_size
		return self.model.meta.feature_size


