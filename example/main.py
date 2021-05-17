#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import cupy
import logging
import numpy as np

MB = 1024**2
chainer.cuda.set_max_workspace_size(256 * MB)
chainer.config.cv_resize_backend = "cv2"

try:
	import matplotlib
except ImportError:
	pass
else:
	matplotlib.use('Agg')

from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from cvfinetune.finetuner import FinetunerFactory
# from cvfinetune.training.trainer import SacredTrainer


from fve_example.core import dataset
from fve_example.core import model as model_module
from fve_example.core import training
from fve_example.utils import parser

def main(args):

	logging.info(f"Chainer version: {chainer.__version__}")

	if args.mode == "visualize":
		raise NotImplementedError

	chainer.set_debug(args.debug)
	if args.debug:
		chainer.config.show()
		cupy.show_config()
		logging.warning("DEBUG MODE ENABLED!")

	tuner_factory = FinetunerFactory.new(args)

	tuner = tuner_factory(opts=args,
		**model_module.get_params(args),
		**dataset.get_params(args),
		**training.updater_params(args),
	)

	logging.info("Profiling the image processing: ")
	with tuner.train_data.enable_img_profiler():
		data = tuner.train_data
		data[np.random.randint(len(data))]

	if args.mode == "train":
		tuner.run(opts=args, **training.trainer_params(args))
	else:
		raise NotImplementedError(f"mode not implemented: {args.mode}")





main(parser.parse_args())
