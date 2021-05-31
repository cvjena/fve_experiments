#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import cv2
import gc
cv2.setNumThreads(0)
import chainer
import cupy
import logging
import numpy as np

MB = 1024**2
chainer.cuda.set_max_workspace_size(256 * MB)
chainer.config.cv_resize_backend = "PIL"

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


def run_profiler(args, tuner):

		from chainer.function_hooks import TimerHook
		from chainer.function_hooks import CupyMemoryProfileHook

		timer_hook = TimerHook()
		memory_hook = CupyMemoryProfileHook()

		with timer_hook, memory_hook:
			args.epochs = 2
			args.no_sacred = True
			args.no_snapshot = True
			tuner.run(opts=args, **training.trainer_params(args, tuner))

		timer_hook.print_report()
		memory_hook.print_report()

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

	train_data = tuner.train_data
	if isinstance(train_data, chainer.datasets.SubDataset):
		train_data = train_data._dataset

	logging.info("Profiling the image processing: ")
	with train_data.enable_img_profiler():
		train_data[np.random.randint(len(train_data))]

	if args.mode == "train":
		if args.profile:
			return run_profiler(args, tuner)

		if args.loss_scaling > 1:
			tuner.opt.loss_scaling(scale=args.loss_scaling)

		if args.optimizer in ["rmsprop", "adam"]:
			tuner.opt.eps = args.opt_epsilon

		return tuner.run(opts=args, **training.trainer_params(args, tuner))
	else:
		raise NotImplementedError(f"mode not implemented: {args.mode}")





main(parser.parse_args())
