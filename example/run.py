#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import logging

from functools import partial
from pathlib import Path

from cvdatasets import AnnotationType

from chainer_addons.models import ModelType
from chainer_addons.models import PrepareType

from fve_example import parser
from fve_example.classifier import Classifier
from fve_example.dataset import new_iterators
from fve_example.training import Trainer



def main(args):

	annot = AnnotationType.new_annotation(args, load_strict=False)
	model_info = annot.info.MODELS[args.model_type]

	model = ModelType.new(
		model_type=model_info.class_key,
		input_size=args.input_size,
		pooling="g_avg"
	)

	size = model.meta.input_size
	prepare = partial(PrepareType[args.prepare_type](model),
		swap_channels=args.swap_channels,
		keep_ratio=True)

	logging.info(" ".join([
		f"Created {model.__class__.__name__} ({args.model_type}) model",
		f"with \"{args.prepare_type}\" prepare function.",
		f"Image input size: {size}",
	]))

	train_it, val_it = new_iterators(args,
		annot=annot,
		prepare=prepare,
		size=size)

	clf = Classifier(args, model=model, annot=annot)

	trainer = Trainer.new(args, clf, train_it, val_it)

	trainer.run()
	# exit(-1)


chainer.config.cv_resize_backend = "cv2"
main(parser.parse_args())
