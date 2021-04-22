from fve_example.core.model.classifier import Classifier
from fve_example.core.model.classifier import FeatureAugmentClassifier



def get_classifier(args):
	if args.augment_features:
		return FeatureAugmentClassifier

	else:
		return Classifier


__all__ = [
	"get_classifier",
	"Classifier",
	"FeatureAugmentClassifier",
]

