import chainer
import chainer.functions as F

from cvmodelz import classifiers
from functools import partial

from fve_fgvc import utils
from fve_fgvc.core.model.classifier.base import BaseFVEClassifier

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