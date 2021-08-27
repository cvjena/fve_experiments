#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import logging
import numpy as np
import matplotlib.pyplot as plt

from os.path import join

from chainer.serializers import save_npz

from fve_fgvc.core import model as model_module
from fve_fgvc.core import dataset as ds_module
from fve_fgvc.core import training
from fve_fgvc.utils import parser


def main(args):
	logging.info(f"Chainer version: {chainer.__version__}")

	chainer.set_debug(args.debug)
	if args.debug:
		logging.warning("DEBUG MODE ENABLED!")

	train, test = ds_module.new_datasets(args)

	if args.mode == "show":
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

	elif args.mode == "train":
		train_it, test_it = ds_module.new_iterators(args, train, test)
		model = model_module.new(args)
		clf = model_module.wrap(model, args)

		trainer = training.setup(args, clf, train_it, test_it)

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

chainer.config.cv_resize_backend = "cv2"
chainer.cuda.set_max_workspace_size(256 * 1024**2)
main(parser.parse_args())
