import chainer
import chainer.functions as F
import chainer.links as L

from cvmodelz import classifiers
from functools import partial

from fve_example import utils
from fve_example.core.model.classifier.base import BaseFVEClassifier


class PartsClassifier(BaseFVEClassifier, classifiers.SeparateModelClassifier):

	@classmethod
	def kwargs(cls, opts) -> dict:
		kwargs = super().kwargs(opts)
		kwargs["copy_mode"] = opts.copy_mode
		kwargs["pred_comb"] = opts.pred_comb
		return kwargs

	def __init__(self, *args, pred_comb, **kwargs):
		super().__init__(*args, **kwargs)

		with self.init_scope():
			self.add_persistent("pred_comb", pred_comb)

	def post_load_init(self):
		super().post_load_init()

		# self.n_classes depends on self.clf shape
		# and it is set proprely only after loading
		if self.pred_comb == "linear":
			self.comb_clf = L.Linear(self.n_classes)

	def load(self, weights: str, n_classes: int, *, finetune: bool = False, **kwargs) -> None:
		if not finetune:
			self.model.reinitialize_clf(self.n_classes, self.model.meta.feature_size)
			self.separate_model.reinitialize_clf(self.n_classes, self.output_size)

		super().load(weights, n_classes, finetune=finetune, **kwargs)

	def load_model(self, *args, finetune: bool = False, **kwargs):
		super().load_model(*args, finetune=finetune, **kwargs)

		if finetune:
			if self.copy_mode == "share":
				clf_name = self.model.clf_layer_name
				new_clf = L.Linear(self.model.meta.feature_size, self.n_classes)
				setattr(self.model, clf_name, new_clf)

			self.model.reinitialize_clf(self.n_classes, self.model.meta.feature_size)


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

		if combined_pred is None:
			return (g_loss + p_loss) * 0.5

		#### This one was used previously,
		#### but does not make sence mathematically
		# 2048 N*2048 -> (N+1)*2048
		# 2048 2*K*D -> 2048 + 2*K*D

		comb_loss = _g_loss(combined_pred)
		return ((g_loss + p_loss) * 0.5 + comb_loss) * 0.5

	def evaluations(self, global_preds, part_preds, combined_pred, aux_preds=None, *, y) -> dict:

		global_accu = self.model.accuracy(global_preds, y)
		part_accu = self.separate_model.accuracy(part_preds, y)

		if combined_pred is None:
			combined_pred = F.log_softmax(global_preds) + F.log_softmax(part_preds)

		evals = {}
		if aux_preds is not None:
			evals["aux_p_accu"] = F.accuracy(aux_preds, y)

			#### This one was used previously,
			#### but does not make sence mathematically
			# part_preds = self.aux_lambda * aux_preds + (1 - self.aux_lambda) * part_preds

			aux_accu = F.accuracy(combined_pred + F.log_softmax(aux_preds), y)
			evals["aux_accu"] = aux_accu

		accu = F.accuracy(combined_pred, y)

		return dict(evals, accu=accu, g_accu=global_accu, p_accu=part_accu)


	@utils.tuple_return
	def predict(self, global_feats, part_feats):
		global_pred = self.model.clf_layer(global_feats)
		part_pred = self.separate_model.clf_layer(part_feats)

		if getattr(self, "comb_clf", None) is not None:
			assert self.pred_comb == "linear"
			comb_feat = F.concat([global_feats, part_feats], axis=1)
			combined_pred = self.comb_clf(comb_feat)

		elif self.pred_comb == "no":
			combined_pred = None #global_pred + part_pred

		elif self.pred_comb == "sum":
			combined_pred = global_pred + part_pred

		else:
			raise ValueError(f"Unknown prediction combination: {self.pred_comb}")

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
		glob_feats, = super().extract(X, self.model)
		part_feats, = super().extract(parts, self.separate_model)

		return glob_feats, part_feats


	def _get_features(self, X, model):
		if X.ndim == 4:
			return super()._get_features(X, model)
		assert X.ndim == 5, f"Malformed input: {X.shape=}"

		n, t, cin, hin, win = X.shape
		_X = X.reshape(n*t, cin, hin, win)
		res = super()._get_features(_X, model)
		_, cout, hout, wout = res.shape
		return res.reshape(n, t, cout, hout, wout)


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
