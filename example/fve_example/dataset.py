import abc
import logging

from functools import partial

from chainer_addons.dataset import AugmentationMixin
from chainer_addons.dataset import PreprocessMixin

from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import CroppedPartMixin
from cvdatasets.utils import new_iterator

class _unpack(abc.ABC):
	def get_example(self, i):
		im_obj = super(_unpack, self).get_example(i)
		im, _, lab = im_obj.as_tuple()
		crops = [] # im_obj.visible_crops(None)
		# ims = crops + [im_obj.array]
		return im, lab

class Dataset(
	AugmentationMixin,
	PreprocessMixin,
	_unpack,
	CroppedPartMixin,
	AnnotationsReadMixin):
	label_shift = None

	def get_example(self, i):
		im, lab = super(Dataset, self).get_example(i)

		# normalize to 0..1
		im -= im.min()
		im /= (im.max() or 1)

		# 0..1 -> -1..1
		im = im * 2 - 1

		return im, lab - self.label_shift

def new_dataset(annot, prepare, size, subset, augment=False):
	kwargs = dict(
		subset=subset,
		preprocess=prepare,
		augment=augment,
		size=size,
		center_crop_on_val=True
	)

	ds = annot.new_dataset(dataset_cls=Dataset, **kwargs)

	logging.info(f"Loaded {len(ds)} images")
	logging.info("Data augmentation is {}abled".format(" en" if augment else "dis"))

	return ds



def new_iterators(args, annot, prepare, size):

	Dataset.label_shift = args.label_shift

	train_data = new_dataset(annot, prepare, size,
		subset="train", augment=True)

	val_data = new_dataset(annot, prepare, size,
		subset="test", augment=False)

	it_kwargs = dict(n_jobs=args.n_jobs, batch_size=args.batch_size)

	train_it, n_batches = new_iterator(train_data,
		**it_kwargs)
	val_it, n_val_batches = new_iterator(val_data,
		repeat=False, shuffle=False, **it_kwargs)

	return train_it, val_it
