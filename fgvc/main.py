#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")
# ruff: noqa: E402

import cv2
cv2.setNumThreads(0)
import chainer
import warnings
import cupy
import logging
import numpy as np

try:
	import PyQt5  # noqa: F401
except ImportError:
	import matplotlib
	matplotlib.use('Agg')
	warnings.warn("PyQt5 was not found, hence, matplotlib's backend was set to 'Agg'!")
else:
	pass
finally:
	pass

from cvfinetune.finetuner import FinetunerFactory
from cvfinetune.parser.utils import populate_args
from pathlib import Path


from fve_fgvc.core import dataset
from fve_fgvc.core import model as model_module
from fve_fgvc.core import training
from fve_fgvc.utils import parser


def run_profiler(args, tuner):

		from chainer.function_hooks import TimerHook
		from chainer.function_hooks import CupyMemoryProfileHook

		timer_hook = TimerHook()
		memory_hook = CupyMemoryProfileHook()

		with timer_hook, memory_hook:
			args.epochs = 2
			args.no_sacred = True
			args.no_snapshot = True
			tuner.run(opts=args, **training.trainer_params())

		timer_hook.print_report()
		memory_hook.print_report()


def main(args):

	if args.mode == "evaluate":
		populate_args(args,
			ignore=[
				"mode", "load", "gpu",
				"mpi", "n_jobs", "batch_size",
				"center_crop_on_val",
				"only_klass",
			],
			replace=dict(
				fve_type={False: "no"},
				pred_comb={False: "no"},
			),
			fc_params=[
				"model/fc/b",
				"model/fc6/b",
				"model/wrapped/output/fc/b",
			]
		)

	logging.info(f"Chainer version: {chainer.__version__}")

	if args.mode == "visualize":
		raise NotImplementedError

	args.dtype = np.empty(0, dtype=chainer.get_dtype()).dtype.name
	logging.info(f"Default dtype: {args.dtype}")

	chainer.set_debug(args.debug)
	if args.debug:
		chainer.config.show()
		cupy.show_config()
		logging.warning("DEBUG MODE ENABLED!")

	tuner_factory = FinetunerFactory()

	tuner = tuner_factory(opts=args,
		experiment_name="FVE Layer",
		manual_gc=True,
		**model_module.get_params(args),
		**dataset.get_params(args),
		**training.updater_params(args),
	)
	tuner.profile_images()

	if args.mode == "train":
		if args.profile:
			return run_profiler(args, tuner)

		if args.loss_scaling > 1:
			tuner.opt.loss_scaling(scale=args.loss_scaling)

		if args.optimizer in ["rmsprop", "adam"]:
			tuner.opt.eps = args.opt_epsilon

		return tuner.run(opts=args, **training.trainer_params())

	elif args.mode == "evaluate":
		dest_folder = Path(args.load).parent
		eval_fname = dest_folder / "evaluation.yml"
		tuner.evaluate(eval_fname, force=args.force)

	else:
		raise NotImplementedError(f"mode not implemented: {args.mode}")



MB = 1024**2
chainer.cuda.set_max_workspace_size(512 * MB)
chainer.config.cv_resize_backend = "cv2"
main(parser.parse_args())
