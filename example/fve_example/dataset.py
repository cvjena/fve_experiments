import abc
import chainer
import logging
import numpy as np

from chainercv import transforms as tr
from contextlib import contextmanager
from functools import partial
from functools import wraps

from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import TransformMixin
from cvdatasets.utils import new_iterator
from cvdatasets.utils import transforms as tr2

def cached(func):

	@wraps(func)
	def inner(self, im_obj):
		key = im_obj._im_path

		if self._cache is not None and key in self._cache:
			return self._cache[key]

		res = func(self, im_obj)

		if self._cache is not None:
			self._cache[key] = res

		return res

	return inner

class Dataset(TransformMixin, AnnotationsReadMixin):
	label_shift = None

	def __init__(self, prepare,
		center_crop_on_val=True, swap_channels=True,
		color_jitter_range=(0, 1),
		*args, **kwargs):
		super(Dataset, self).__init__(*args, **kwargs)
		self.prepare = prepare
		if isinstance(prepare, partial):
			assert swap_channels == prepare.keywords.get("swap_channels"), \
				("swap_channels options was different in the prepare function "
				"and the preprocessing!")

		self.center_crop_on_val = center_crop_on_val
		self.channel_order = "BGR" if swap_channels else "RGB"

		self.min_value, self.max_value = color_jitter_range

		self._cache = None #{}
		self._profile_img_enabled = False

	@contextmanager
	def enable_img_profiler(self):
		_dmp = self._profile_img_enabled
		self._profile_img_enabled = True
		yield
		self._profile_img_enabled = _dmp

	def _profile_img(self, img, tag):
		if self._profile_img_enabled:
			print(f"[{tag:^20s}]"
				" | ".join([
					f"size: {str(img.shape):>20s}",
					f"pixel values: ({img.min():+8.2f}, {img.max():+8.2f})"
					])
				)
		return img

	@property
	def augmentations(self):

		if chainer.config.train:
			return [
				(tr.random_crop, dict(size=self._size)),
				(tr.random_flip, dict(x_random=True, y_random=True)),
				(tr.random_rotate, dict()),
				(tr2.color_jitter, dict(
					brightness=0.1,
					contrast=0.1,
					saturation=0.1,
					channel_order=self.channel_order,
					min_value=self.min_value,
					max_value=self.max_value,
				))
			]

		else:
			if self.center_crop_on_val:
				return [
					(tr.center_crop, dict(size=self.size)),
				]

			else:
				return []

	@cached
	def preprocess(self, im_obj):

		im, _, lab = im_obj.as_tuple()
		lab -= self.label_shift

		if self._annot.part_type == "GLOBAL":
			ims = []

		else:
			ims = im_obj.visible_crops(None)

		ims.insert(0, im)

		res = []
		for im in ims:
			self._profile_img(im, "before prepare")

			im = self.prepare(im, size=self.size)
			self._profile_img(im, "prepare")

			res.append(im)

		return res, lab

	def augment(self, ims):
		res = []
		for im in ims:
			for aug, params in self.augmentations:
				im = self._profile_img(aug(im, **params), aug.__name__)

			res.append(im)

		return res

	def postprocess(self, ims):
		ims = np.array(ims)
		if self.max_value == 1:
			# 0..1 -> -1..1
			ims = ims * 2 - 1

		self._profile_img(ims, "postprocess")
		# leave as they are
		return ims

	def transform(self, im_obj):

		ims, lab = self.preprocess(im_obj)
		ims = self.augment(ims)
		ims = self.postprocess(ims)

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

def new_dataset(annot, subset, **kwargs):
	ds = annot.new_dataset(dataset_cls=Dataset, subset=subset, **kwargs)
	logging.info(f"Loaded {len(ds)} images")
	return ds



def new_iterators(args, annot, prepare, size):

	Dataset.label_shift = args.label_shift
	color_jitter_range = (None, None) if args.model_type == "resnet" else (0, 1)

	ds_kwargs = dict(
		prepare=prepare,
		size=size,
		swap_channels=args.swap_channels,
		color_jitter_range=color_jitter_range,
	)

	train_data = new_dataset(annot, subset="train", **ds_kwargs)

	logging.info(f"Profiling image processing... ")
	with train_data.enable_img_profiler():
		train_data[0]

	val_data = new_dataset(annot, subset="test", **ds_kwargs)

	it_kwargs = dict(
		n_jobs=args.n_jobs,
		batch_size=args.batch_size,
	)

	train_it, n_batches = new_iterator(train_data, **it_kwargs)
	val_it, n_val_batches = new_iterator(val_data,
		repeat=False, shuffle=False, **it_kwargs)

	return train_it, val_it
