import abc
import chainer
import numpy as np
import logging

from functools import partial
from chainercv import transforms as tr

from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import TransformMixin
from cvdatasets.utils import new_iterator
from cvdatasets.utils import transforms as tr2

class Dataset(TransformMixin, AnnotationsReadMixin):
	label_shift = None

	def __init__(self, prepare, center_crop_on_val=True, *args, **kwargs):
		super(Dataset, self).__init__(*args, **kwargs)
		self.prepare = prepare
		self.center_crop_on_val = center_crop_on_val

	@property
	def augmentations(self):

		if chainer.config.train:
			return [
				(tr.random_crop, dict(size=self._size)),
				(tr.random_flip, dict(x_random=True, y_random=True)),
				(tr.random_rotate, dict()),
				(tr2.color_jitter, dict(
					brightness=0.4,
					contrast=0.4,
					saturation=0.4,
					max_value=1,
				))
			]

		else:
			if self.center_crop_on_val:
				return [
					(tr.center_crop, dict(size=self.size)),
				]

			else:
				return []

	def preprocess(self, ims):
		res = []
		for im in ims:
			_ = im.shape
			im = self.prepare(im, size=self.size)
			# logging.debug("preprocess:", _, "->", im.shape)
			# normalize to 0..1
			im -= im.min()
			im /= (im.max() or 1)
			res.append(im)

		return res

	def augment(self, ims):
		res = []
		for im in ims:
			for aug, params in self.augmentations:
				_ = im.shape
				im = aug(im, **params)
				# logging.debug(aug.__name__, _, "->", im.shape)
			res.append(im)

		return res

	def postprocess(self, ims):
		ims = np.array(ims)
		# 0..1 -> -1..1
		return ims * 2 - 1

	def transform(self, im_obj):
		im, _, lab = im_obj.as_tuple()
		if self._annot.part_type == "GLOBAL":
			ims = []

		else:
			ims = im_obj.visible_crops(None)

		ims.insert(0, im)

		ims = self.preprocess(ims)
		ims = self.augment(ims)
		ims = self.postprocess(ims)

		lab -= self.label_shift

		return ims, lab

	def _prepare_back(self, im):
		return im.transpose(1,2,0) / 2 + .5

	def get_example(self, i):
		ims, lab = super(Dataset, self).get_example(i)

		glob_im, parts = ims[0], ims[1:]
		if len(parts) == 0:
			return glob_im, lab

		else:
			return glob_im, parts, lab

def new_dataset(annot, prepare, size, subset):
	kwargs = dict(
		subset=subset,
		prepare=prepare,
		size=size,
	)

	ds = annot.new_dataset(dataset_cls=Dataset, **kwargs)

	logging.info(f"Loaded {len(ds)} images")

	return ds



def new_iterators(args, annot, prepare, size):

	Dataset.label_shift = args.label_shift

	train_data = new_dataset(annot, prepare, size, subset="train")

	val_data = new_dataset(annot, prepare, size, subset="test")

	it_kwargs = dict(n_jobs=args.n_jobs, batch_size=args.batch_size)

	train_it, n_batches = new_iterator(train_data,
		**it_kwargs)
	val_it, n_val_batches = new_iterator(val_data,
		repeat=False, shuffle=False, **it_kwargs)

	return train_it, val_it
