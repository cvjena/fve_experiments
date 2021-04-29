import logging

from os.path import join

from chainer.backends import cuda
from chainer.training.updaters import StandardUpdater

from chainer_addons.training import MiniBatchUpdater

from cvdatasets.utils import pretty_print_dict

def updater_params(args) -> dict:

	updater_cls = StandardUpdater
	updater_kwargs = dict()

	if args.update_size > args.batch_size:
		updater_cls = MiniBatchUpdater
		updater_kwargs["update_size"] = args.update_size

	return dict(
		updater_cls=updater_cls,
		updater_kwargs=updater_kwargs,
	)

