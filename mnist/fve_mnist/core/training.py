import logging
import numpy as np
import pyaml

from pathlib import Path

from chainer.backends import cuda
from chainer.training import Trainer as DefaultTrainer
from chainer.training import extensions
from chainer.training.updaters import StandardUpdater

from chainer_addons.training import MiniBatchUpdater
from chainer_addons.training import lr_shift
from chainer_addons.training import optimizer

from cvdatasets.utils import attr_dict
from cvdatasets.utils import pretty_print_dict


default_intervals = attr_dict(
	print =		(1,  'epoch'),
	log =		(1,  'epoch'),
	eval =		(1,  'epoch'),
	snapshot =	(10, 'epoch'),
)

def gpu_config(args):
	if -1 in args.gpu:
		return -1

	device_id = args.gpu[0]
	device = cuda.get_device_from_id(device_id)
	device.use()
	return device

def get_updater(args):

	device = gpu_config(args)

	updater_kwargs = dict(device=device)
	if args.update_size > args.batch_size:
		updater_cls = MiniBatchUpdater
		updater_kwargs["update_size"] = args.update_size

	else:
		updater_cls = StandardUpdater


	logging.info(" ".join([
		f"Using single GPU: {device}.",
		f"{updater_cls.__name__} is initialized",
		f"with following kwargs: {pretty_print_dict(updater_kwargs)}"
		])
	)
	return updater_cls, updater_kwargs

def setup_snapshots(trainer, args, obj, trigger):

	if args.no_snapshot:
		logging.warning("Models are not snapshot!")

	else:
		dump_fmt = "ft_model_epoch{0.updater.epoch:03d}.npz"
		# trainer.extend(extensions.snapshot_object(obj, dump_fmt), trigger=trigger)
		# logging.info("Snapshot format: \"{}\"".format(dump_fmt))

def eval_name(evaluator, name):
	if evaluator is None:
		return name

	return f"{evaluator.default_name}/{name}"

def setup(args, model, train_it, val_it=None, intervals=default_intervals):
	outdir = args.output
	logging.info("Training outputs are saved under \"{}\"".format(outdir))

	with open(Path(outdir, "args.yml"), "w") as out:
		pyaml.dump(args.__dict__, out)

	opt_kwargs = {}
	# if args.optimizer == "rmsprop":
	# 	opt_kwargs["alpha"] = 0.9
	opt = optimizer(args.optimizer,
		model=model,
		lr=args.learning_rate,
		decay=args.decay,
		gradient_clipping=False,
		**opt_kwargs)

	# if args.only_clf:
	# 	target.disable_update()
	# 	target.model.clf_layer.enable_update()
	updater_cls, updater_kwargs = get_updater(args)

	train_ds = train_it.dataset
	with train_ds.enable_img_profiler():
		train_ds[np.random.randint(len(train_ds))]

	updater = updater_cls(
		iterator=train_it,
		optimizer=opt,
		**updater_kwargs
	)

	if val_it is None:
		evaluator = None

	else:
		evaluator = extensions.Evaluator(
			iterator=val_it,
			target=model,
			device=updater.device
		)
		evaluator.default_name = "val"


	trainer = DefaultTrainer(
		updater=updater,
		stop_trigger=(args.epochs, "epoch"),
		out=outdir
	)

	if evaluator is not None:
		trainer.extend(evaluator, trigger=intervals.eval)

	lr_shift_ext = lr_shift(opt,
		init=args.learning_rate,
		rate=args.lr_decrease_rate, target=args.lr_target)
	trainer.extend(lr_shift_ext, trigger=(args.lr_shift, 'epoch'))

	trainer.extend(extensions.observe_lr(), trigger=intervals.log)
	trainer.extend(extensions.LogReport(trigger=intervals.log))

	setup_snapshots(trainer, args, model, intervals.snapshot)
	print_values = [
		"elapsed_time",
		"epoch",

		"main/accu", eval_name(evaluator, "main/accu"),
		"main/loss", eval_name(evaluator, "main/loss"),

	]
	trainer.extend(extensions.PrintReport(print_values),
		trigger=intervals.print)

	if not args.no_progress:
		trainer.extend(extensions.ProgressBar(update_interval=100))

	return trainer


