import logging

from os.path import join

from chainer.backends import cuda
from chainer.training.updaters import StandardUpdater

from chainer_addons.training import MiniBatchUpdater

from cvdatasets.utils import pretty_print_dict


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

