#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import cupy
import logging

from functools import partial

from cvdatasets import AnnotationType
from cvdatasets.dataset.image import Size

from chainer_addons.models import ModelType
from chainer_addons.models import PrepareType

from fve_example import parser
from fve_example.classifier import Classifier
from fve_example.dataset import new_iterators
from fve_example.training import Trainer
from fve_example.visualizer import Visualizer



def main(args):
	print(f"Chainer version: {chainer.__version__}")
	chainer.config.show()
	cupy.show_config()

	annot = AnnotationType.new_annotation(args, load_strict=False)
	model_info = annot.info.MODELS[args.model_type]


	input_size = Size(args.input_size)
	model = ModelType.new(
		model_type=model_info.class_key,
		input_size=input_size,
		pooling="g_avg"
	)

	prepare = partial(PrepareType[args.prepare_type](model),
		swap_channels=args.swap_channels,
		keep_ratio=True)

	logging.info(" ".join([
		f"Created {model.__class__.__name__} ({args.model_type}) model",
		f"with \"{args.prepare_type}\" prepare function.",
		f"Image input size: {input_size}",
	]))

	train_it, val_it = new_iterators(args,
		annot=annot,
		prepare=prepare,
		size=input_size)

	if args.mode == "train":
		clf = Classifier.new(args, model=model, annot=annot)

		trainer = Trainer.new(args, clf, train_it, val_it)
		trainer.run()

	elif args.mode == "visualize":
		clf = Classifier.load(args, model=model, annot=annot)

		it = train_it if args.subset == "train" else val_it
		vis = Visualizer.new(args, clf, it.dataset)
		vis.run()

	else:
		raise ValueError(f"Unknown mode: {args.mode}")
	# exit(-1)


chainer.config.cv_resize_backend = "cv2"
main(parser.parse_args())
