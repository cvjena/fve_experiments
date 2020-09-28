import chainer
import chainer.functions as F
import chainer.links as L
import logging

from functools import partial
from os.path import join

from chainer_addons.functions import smoothed_cross_entropy

from fve_layer.backends.chainer.links import FVELayer
from fve_layer.backends.chainer.links import FVELayer_noEM

def _unpack(var):
	return var[0] if isinstance(var, tuple) else var

class Classifier(chainer.Chain):


	def __init__(self, args, model, annot):
		super(Classifier, self).__init__()


		info = annot.info
		model_info = info.MODELS[args.model_type]
		n_classes = info.DATASETS[args.dataset].n_classes

		default_weights = join(
			info.BASE_DIR,
			info.MODEL_DIR,
			model_info.folder,
			model_info.weights
		)

		with self.init_scope():
			self.model = model
			self.init_encoding(args)
			self.init_aux_clf(args, n_classes)

		self._load_weights(args, default_weights, n_classes)
		self._init_loss(args, n_classes)


	def report(self, **values):
		chainer.report(values, self)

	def init_aux_clf(self, args, n_classes):
		if args.aux_lambda > 0:
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
				em=FVELayer,
				grad=FVELayer_noEM
			)
			assert args.fve_type in fve_classes, \
				f"Unknown FVE type: {args.fve_type}!"

			fve_class = fve_classes[args.fve_type]
			logging.info(f"=== Using {fve_class.__name__} ({args.fve_type}) FVE-layer ===")

			self.fve_layer = fve_class(
				in_size=args.comp_size,
				n_components=args.n_components
			)

			self._output_size = 2 * fv_insize * n_comps

		logging.info(f"Final pre-classification size: {self.output_size}")
		self.add_persistent("aux_lambda", args.aux_lambda)
		self.add_persistent("mask_features", args.mask_features)

	def encode(self, feats):

		if self.fve_layer is None:
			return F.mean(feats, axis=1)

		n, t, c, h, w = feats.shape

		# N x T x C x H x W -> N x T x H x W x C
		feats = F.transpose(feats, (0, 1, 3, 4, 2))
		# N x T x H x W x C -> N x T*H*W x C
		feats = F.reshape(feats, (n, t*h*w, c))

		logits = self.fve_layer(feats, use_mask=self.mask_features)

		return logits

	def _get_conv_map(self, x):
		return _unpack(self.model(x,
					layer_name=self.model.meta.conv_map_layer))

	def extract_global(self, X):
		conv_map = self._get_conv_map(X)

		if self.fve_layer is None:
			return self.model.pool(conv_map)

		feats = self.pre_fve(conv_map)
		feats = F.expand_dims(feats, axis=1)

		return self.encode(feats)


	def extract_parts(self, parts):
		if parts is None:
			return None

		part_convs = []
		_pre_fve = self.model.pool if self.pre_fve is None else self.pre_fve

		for part in parts.transpose(1,0,2,3,4):
			part_conv_map = self._get_conv_map(part)
			part_conv_map = _pre_fve(part_conv_map)
			part_convs.append(part_conv_map)

		# store it for the auxilary classifier
		self.part_convs = part_convs = F.stack(part_convs, axis=1)

		return self.encode(part_convs)

	def extract(self, X, parts=None):
		return self.extract_global(X), self.extract_parts(parts)

	def predict_global(self, y, global_logit):
		pred = self.model.clf_layer(global_logit)

		self.report(
			glob_accu=F.accuracy(pred, y),
			glob_loss=self.loss(pred, y)
		)

		return pred


	def predict_parts(self, y, part_logits):
		if part_logits is None:
			return None

		pred = self.model.clf_layer(part_logits)

		self.report(
			part_accu=F.accuracy(pred, y),
			part_loss=self.loss(pred, y)
		)

		if not hasattr(self, "aux_clf"):
			self.part_convs = None
			return pred

		n, t, c, h, w = self.part_convs.shape
		map_feats = F.mean(self.part_convs, axis=(3,4))

		_map_feats = F.reshape(map_feats, shape=(n*t, c))
		_aux_pred = self.aux_clf(_map_feats)
		aux_pred = F.reshape(_aux_pred, shape=(n, t, -1))
		aux_pred = F.sum(aux_pred, axis=1)

		return pred * (1 - self.aux_lambda) + aux_pred * self.aux_lambda

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

		self.report(accuracy=accu, loss=loss)
		return loss


	def forward(self, *inputs):
		assert len(inputs) in [2, 3], \
			(f"Expected 2 (image and label) or 3 (image, parts, and label) inputs,"
			"got {len(inputs)}!")

		*X, y = inputs

		logits = self.extract(*X)
		preds = self.predict(y, *logits)
		return self.get_loss(y, *preds)


		return loss



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


	def _load_weights(self, args, weights, n_classes):

		if args.from_scratch:
			logging.info("No weights loaded, training from scratch!")
			self.model.reinitialize_clf(n_classes=n_classes, feat_size=self.output_size)
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

		logging.info(msg)
		loader(
			weights=weights,
			n_classes=n_classes,
			feat_size=self.output_size,

			strict=args.load_strict,
			headless=args.headless,
		)


	@property
	def output_size(self):
		return self._output_size
		return self.model.meta.feature_size

