from cvmodelz import classifiers

from fve_example.core.model.classifier import FVEMixin
from fve_example.core.model.classifier import Classifier


def get_classifier(opts):
	if opts.augment_features:
		raise NotImplementedError
		return FeatureAugmentClassifier

	else:
		return Classifier


def get_params(opts) -> dict:
	clf_cls = get_classifier(opts)
	clf_kwargs = dict(
		only_head=opts.only_head,
		**FVEMixin.kwargs(opts)
	)

	return dict(
		classifier_cls=clf_cls,
		classifier_kwargs=clf_kwargs,

		model_kwargs=dict(pooling="g_avg")
	)


__all__ = [
	"get_classifier",
	"Classifier",
]

