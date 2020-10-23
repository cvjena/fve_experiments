#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import cupy
import logging

from os.path import join
from functools import partial
from chainercv2.model_provider import get_model
from chainercv2.models import model_store

from cvdatasets import AnnotationType
from cvdatasets.dataset.image import Size

from chainer_addons.models import ModelType
from chainer_addons.models import PrepareType

from fve_example import parser
from fve_example.classifier import Classifier
from fve_example.classifier import ModelWrapper
from fve_example.dataset import new_iterators
from fve_example.training import Trainer
from fve_example.visualizer import Visualizer


def main(args):
	print(f"Chainer version: {chainer.__version__}")
	# chainer.config.show()
	# cupy.show_config()

	annot = AnnotationType.new_annotation(args, load_strict=False)
	info = annot.info
	ds_info = info.DATASETS[args.dataset]

	input_size = Size(args.input_size)
	if args.model_type.startswith("cv2_"):

		model_type = args.model_type.split("cv2_")[-1]
		model = get_model(model_type, pretrained=False)
		model = ModelWrapper(model)
		model.meta.input_size = input_size

		args.prepare_type = PrepareType.CHAINERCV2.name.lower()
		prepare = PrepareType.CHAINERCV2(model)
		default_weights = model_store.get_model_file(
			model_name=model_type,
			local_model_store_dir_path=join("~", ".chainer", "models"))

	else:
		model_info = annot.info.MODELS[args.model_type]
		default_weights = join(
			info.BASE_DIR,
			info.MODEL_DIR,
			model_info.folder,
			model_info.weights
		)

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
		clf = Classifier.new(args,
			n_classes=ds_info.n_classes,
			model=model,
			default_weights=default_weights)

		trainer = Trainer.new(args, clf, train_it, val_it)
		trainer.run()

	elif args.mode == "visualize":
		clf = Classifier.load(args,
			n_classes=ds_info.n_classes,
			model=model,
			default_weights=default_weights)

		it = train_it if args.subset == "train" else val_it
		vis = Visualizer.new(args, clf, it.dataset)
		vis.run()

	else:
		raise ValueError(f"Unknown mode: {args.mode}")
	# exit(-1)


chainer.config.cv_resize_backend = "cv2"
main(parser.parse_args())
