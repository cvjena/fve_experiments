import logging
import chainer

from chainer import functions as F
from chainer import links as L

# from chainer_addons.models import PrepareType

from chainercv2.model_provider import get_model
# from chainercv2.models import model_store


def new(args):
	model = get_model(args.model_type, pretrained=False)
	logging.info(f"Created {model.__class__.__name__} ({args.model_type}) model",)
	# prepare = PrepareType.CHAINERCV2(model)
	# default_weights = model_store.get_model_file(
	# 	model_name=model_type,
	# 	local_model_store_dir_path=join("~", ".chainer", "models"))

	return model

def wrap(model, args):
	return Classifier(model=model)


class Classifier(chainer.Chain):
	def __init__(self, model):
		super(Classifier, self).__init__()

		with self.init_scope():
			self.model = model
			self.post_model = L.Convolution2D(None, 2, ksize=1)

			self.fc = L.Linear(2, self.model.classes)

		delattr(self.model.features, "final_pool")

	def report(self, **values):
		chainer.report(values, self)

	def extract(self, X):
		conv_maps = self.model.features(X)
		feats = self.post_model(conv_maps)
		return feats

	def encode(self, feats):
		return F.mean(feats, axis=(-2,-1))

	def predict(self, logits, y):
		preds = self.fc(logits)
		accu = F.accuracy(preds, y)
		self.report(accu=accu)
		return preds

	def get_loss(self, preds, y):
		loss = F.softmax_cross_entropy(preds, y)
		self.report(loss=loss)
		return loss

	def forward(self, X, y):
		feats = self.extract(X)
		logits = self.encode(feats)

		preds = self.predict(logits, y)

		return self.get_loss(preds, y)

