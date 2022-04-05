#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import cv2
cv2.setNumThreads(0)
import chainer
import cupy
import logging
import yaml
import pyaml
import numpy as np

MB = 1024**2
chainer.cuda.set_max_workspace_size(256 * MB)
chainer.config.cv_resize_backend = "PIL"

# try:
# 	import matplotlib
# except ImportError:
# 	pass
# else:
# 	matplotlib.use('Agg')

from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from cvfinetune.finetuner import FinetunerFactory
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

def populate_args(args,
	ignore=[
		"mode", "load", "gpu",
		"mpi", "n_jobs", "batch_size",
		"center_crop_on_val",
		"only_klass",
		],
	replace=dict(fve_type={False: "no"}, pred_comb={False: "no"}) ):

	args.debug = False

	assert args.load is not None, "--load argument missing!"

	model_path = Path(args.load)

	args_path = model_path.parent / "meta" / "args.yml"

	assert args_path.exists(), f"Couldn't find args file \"{args_path}\""

	logging.info(f"Setting arguments from \"{args_path}\"")

	with open(args_path) as f:
		dumped_args: dict = yaml.safe_load(f)

	for key, value in dumped_args.items():
		if key in ignore or getattr(args, key, None) == value:
			continue

		old_value = getattr(args, key, None)
		if key in replace:
			value = replace[key].get(value, value)

		logging.debug(f"Setting \"{key}\" to {value} (originally was {'missing' if old_value is None else old_value})")

		setattr(args, key, value)

	# get the correct number of classes
	args.n_classes = 1000
	weights = np.load(args.load)
	for key in ["model/fc/b", "model/wrapped/output/fc/b"]:
		try:
			args.n_classes = weights[key].shape[0]
		except KeyError as e:
			pass


def main(args):

	if args.mode == "evaluate":
		populate_args(args)

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

	tuner_factory = FinetunerFactory.new(mpi=args.mpi)

	tuner = tuner_factory(opts=args,
		experiment_name="FVE Layer",
		manual_gc=True,
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

		return tuner.run(opts=args, **training.trainer_params())

	elif args.mode == "evaluate":
		dest_folder = Path(args.load).parent
		eval_fname = dest_folder / "evaluation.yml"
		if eval_fname.exists() and not args.force:
			print(f"Evaluation file exists already, skipping \"{args.load}\"")
			return
		res = tuner.evaluator()
		res = {key: float(value) for key, value in res.items()}
		with open(eval_fname, "w") as f:
			pyaml.dump(res, f, sort_keys=False)
	else:
		raise NotImplementedError(f"mode not implemented: {args.mode}")





main(parser.parse_args())
