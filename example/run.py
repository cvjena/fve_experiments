#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import cupy
import logging

from chainercv2.model_provider import get_model
from chainercv2.models import model_store
from functools import partial
from os.path import join

from cvdatasets import AnnotationType
from cvdatasets.dataset.image import Size

from chainer_addons.models import ModelType
from chainer_addons.models import PrepareType

from fve_example.core import model as model_module
from fve_example.core import dataset
from fve_example.core import training
from fve_example.core import visualizer
from fve_example.utils import parser


def main(args):
	print(f"Chainer version: {chainer.__version__}")

	chainer.set_debug(args.debug)
	if args.debug:
		logging.warning("DEBUG MODE ENABLED!")

	# chainer.config.show()
	# cupy.show_config()

	annot = AnnotationType.new_annotation(args, load_strict=False)
	info = annot.info
	ds_info = info.DATASETS[args.dataset]

	input_size = Size(args.input_size)
	parts_input_size = Size(args.parts_input_size)
	if args.model_type.startswith("cv2_"):

		model_type = args.model_type.split("cv2_")[-1]
		model = get_model(model_type, pretrained=False)
		model = model_module.ModelWrapper(model)
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
		f"Image input size: {input_size}.",
		f"Image parts input size: {parts_input_size}",
	]))

	train_it, val_it = dataset.new_iterators(args,
		annot=annot,
		prepare=prepare,
		size=input_size,
		part_size=parts_input_size
	)

	if args.mode == "train":
		clf = model_module.Classifier.new(args,
			n_classes=ds_info.n_classes,
			model=model,
			default_weights=default_weights)

		trainer = training.Trainer.new(args, clf, train_it, val_it)
		trainer.run()

	elif args.mode == "visualize":
		clf = model_module.Classifier.load(args,
			n_classes=ds_info.n_classes,
			model=model,
			default_weights=default_weights)

		it = train_it if args.subset == "train" else val_it
		vis =  visualizer.Visualizer.new(args, clf, it.dataset)
		vis.run()

	else:
		raise ValueError(f"Unknown mode: {args.mode}")
	# exit(-1)


chainer.config.cv_resize_backend = "cv2"
chainer.cuda.set_max_workspace_size(256 * 1024**2)
main(parser.parse_args())
