#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import logging
import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from bdb import BdbQuit

from chainer.serializers import save_npz

from fve_mnist.core import model as model_module
from fve_mnist.core import dataset as ds_module
from fve_mnist.core import training
from fve_mnist.utils import parser


def show(args):

	try:
		import PyQt5  # noqa: F401
	except ImportError:
		raise RuntimeError("PyQt is not installed, so the visualization does not work! "
			"Please install it with 'conda install pyqt'.")

	train, test = ds_module.new_datasets(args)
	dataset = dict(train=train, test=test).get(args.subset)
	rnd = np.random.RandomState(args.seed)
	idxs = rnd.choice(len(dataset), args.n_samples, replace=False)

	samples = dataset[idxs]

	n_rows = int(np.ceil(np.sqrt(args.n_samples)))
	n_cols = int(np.ceil(args.n_samples / n_rows))

	fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)
	for i, (im, lab) in enumerate(samples):
		im = im.transpose(1,2,0)
		ax = axs[np.unravel_index(i, axs.shape)]
		ax.imshow(im)
		ax.axis("off")
		ax.set_title(str(lab))

	plt.tight_layout()
	plt.show()
	plt.close()

def train(args):
	train, test = ds_module.new_datasets(args, n_samples=1000)
	train_it, test_it = ds_module.new_iterators(args, train, test)
	model = model_module.new(args)
	clf = model_module.wrap(model, args)

	trainer = training.setup(args, clf, train_it, test_it,
		progress_update=10)

	logging.info("Snapshotting is {}abled".format("dis" if args.no_snapshot else "en"))

	def dump(suffix):
		if args.no_snapshot:
			return

		save_npz(join(trainer.out,
			"clf_{}.npz".format(suffix)), clf)
		# save_npz(join(trainer.out,
		# 	"model_{}.npz".format(suffix)), model)

	try:
		trainer.run()
	except (KeyboardInterrupt, BdbQuit) as e:
		raise e
	except Exception as e:
		dump("exception")
		raise e
	else:
		dump("final")

def main(args):
	logging.info(f"Chainer version: {chainer.__version__}")

	chainer.set_debug(args.debug)
	if args.debug:
		logging.warning("DEBUG MODE ENABLED!")

	if args.mode == "show":
		show(args)

	elif args.mode == "train":
		train(args)


chainer.config.cv_resize_backend = "cv2"
chainer.cuda.set_max_workspace_size(512 * 1024**2)
main(parser.parse_args())
