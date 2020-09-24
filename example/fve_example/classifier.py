import chainer
import chainer.functions as F
import logging

from functools import partial
from os.path import join

from chainer_addons.functions import smoothed_cross_entropy

class Classifier(chainer.Chain):


	def __init__(self, args, model, annot):
		super(Classifier, self).__init__()

		with self.init_scope():
			self.model = model

		info = annot.info
		model_info = info.MODELS[args.model_type]
		ds_info = info.DATASETS[args.dataset]
		default_weights = join(
			info.BASE_DIR,
			info.MODEL_DIR,
			model_info.folder,
			model_info.weights
		)
		self._load_weights(args, default_weights, ds_info.n_classes)
		self._init_loss(args, ds_info.n_classes)


	def forward(self, *inputs):
		assert len(inputs) == 2, \
			f"Expected 2 inputs (image and label), got {len(inputs)}!"

		X, y = inputs

		pred = self.model(X)

		if isinstance(pred, tuple):
			pred = pred[0]

		loss = self.loss(pred, y)
		accu = F.accuracy(pred, y)

		chainer.report(dict(accuracy=accu, loss=loss), self)

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
		return self.model.meta.feature_size

