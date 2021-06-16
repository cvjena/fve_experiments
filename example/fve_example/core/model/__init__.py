from cvmodelz import classifiers

from fve_example.core.model.classifier import GlobalClassifier
from fve_example.core.model.classifier import PartsClassifier


def get_classifier(opts):
	if opts.parts == "GLOBAL":
		return GlobalClassifier

	else:
		return PartsClassifier


def get_params(opts) -> dict:
	clf_cls = get_classifier(opts)
	clf_kwargs = dict(
		only_head=opts.only_head,
		**clf_cls.kwargs(opts)
	)

	model_kwargs = dict(pooling="g_avg")

	if hasattr(opts, "n_classes"):
		model_kwargs["n_classes"] = opts.n_classes

	return dict(
		classifier_cls=clf_cls,
		classifier_kwargs=clf_kwargs,

		model_kwargs=model_kwargs
	)


__all__ = [
	"get_classifier",
	"GlobalClassifier",
	"PartsClassifier",
]

