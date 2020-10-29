import abc
import chainer
import logging
import numpy as np

from chainercv import transforms as tr
from contextlib import contextmanager
from functools import partial
from functools import wraps

from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import BasePartMixin
from cvdatasets.dataset import TransformMixin
from cvdatasets.utils import new_iterator
from cvdatasets.utils import transforms as tr2


def new_dataset(annot, subset, **kwargs):
	ds = annot.new_dataset(dataset_cls=Dataset, subset=subset, **kwargs)
	logging.info(f"Loaded {len(ds)} images")
	return ds


def new_iterators(args, annot, prepare, size, part_size=None):

	Dataset.label_shift = args.label_shift
	color_jitter_range = (None, None) if args.model_type == "resnet" else (0, 1)
	part_size = size if part_size is None else part_size

	ds_kwargs = dict(
		prepare=prepare,
		size=size,
		part_size=part_size,
		opts=args,
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

class Dataset(TransformMixin, BasePartMixin, AnnotationsReadMixin):
	label_shift = None

	def __init__(self, prepare, opts, *args, **kwargs):
		super(Dataset, self).__init__(*args, **kwargs)
		self.prepare = prepare
		self._cache = None #{} if opts.cache_images else None
		self._profile_img_enabled = False

		self._setup_augmentations(opts)

		# for these models, we need to scale from 0..1 to -1..1
		self.zero_mean = opts.model_type in ["inception", "inception_imagenet"]


	def _setup_augmentations(self, opts):

		min_value, max_value = (0, 1) if opts.model_type in ["inception", "inception_imagenet"] else (None, None)

		pos_augs = dict(
			random_crop=(tr.random_crop, dict(size=self._size)),

			center_crop=(tr.center_crop, dict(size=self._size)),

			random_flip=(tr.random_flip, dict(x_random=True, y_random=False)),

			random_rotate=(tr.random_rotate, dict()),

			color_jitter=(tr2.color_jitter, dict(
				brightness=opts.brightness_jitter,
				contrast=opts.contrast_jitter,
				saturation=opts.saturation_jitter,
				channel_order="BGR" if opts.swap_channels else "RGB",
				min_value=min_value,
				max_value=max_value,
			)),

		)

		logging.info("Enabled following augmentations in the training phase: " + ", ".join(opts.augmentations))

		self._train_augs = list(map(pos_augs.get, opts.augmentations))
		self._val_augs = []

		if opts.center_crop_on_val:
			logging.info("During evaluation, center crop is used!")
			self._val_augs.append(pos_augs["center_crop"])


	@contextmanager
	def enable_img_profiler(self):
		_dmp = self._profile_img_enabled
		self._profile_img_enabled = True
		yield
		self._profile_img_enabled = _dmp

	def _profile_img(self, img, tag):
		if len(img) == 0: return
		if self._profile_img_enabled:
			print(f"[{tag:^20s}]"
				" | ".join([
					f"size: {str(img.shape):>20s}",
					f"pixel values: ({img.min():+8.2f}, {img.max():+8.2f})"
					])
				)

	@property
	def augmentations(self):
		return self._train_augs if chainer.config.train else self._val_augs

	def preprocess_parts(self, im_obj):

		if self._annot.part_type == "GLOBAL":
			return []

		parts = []
		for i, part in enumerate(im_obj.visible_crops(self.ratio)):
			if i == 0:
				self._profile_img(part, "(part) before prepare")

			part = self.prepare(part, size=self.part_size)

			if i == 0:
				self._profile_img(part, "(part) prepare")

			parts.append(part)

		return parts

	@cached
	def preprocess(self, im_obj):

		im, _, lab = im_obj.as_tuple()

		self._profile_img(im, "before prepare")
		im = self.prepare(im, size=self.size)
		self._profile_img(im, "prepare")

		lab -= self.label_shift
		parts = self.preprocess_parts(im_obj)

		return im, parts, lab

	def augment_parts(self, parts, profile=True):
		res = []
		for i, part in enumerate(parts):
			for aug, params in self.augmentations:

				if "size" in params:
					params = dict(params)
					params["size"] = self._part_size

				part = aug(part, **params)

				if i == 0:
					self._profile_img(part, aug.__name__)

			res.append(part)
		return res


	def augment(self, im, parts):

		for aug, params in self.augmentations:
			im = aug(im, **params)
			self._profile_img(im, aug.__name__)

		aug_parts = self.augment_parts(parts)

		return im, aug_parts

	def postprocess(self, im, parts):

		parts = np.array(parts, dtype=im.dtype)
		if self.zero_mean:
			# 0..1 -> -1..1
			im = im * 2 - 1
			parts = parts * 2 - 1

		self._profile_img(im, "postprocess")
		self._profile_img(parts, "(parts) postprocess")

		# leave as they are
		return im, parts

	def transform(self, im_obj):

		im, parts, lab = self.preprocess(im_obj)
		im, parts = self.augment(im, parts)
		im, parts = self.postprocess(im, parts)

		if len(parts) == 0:
			return im, lab

		else:
			return im, parts, lab

	def _prepare_back(self, im):
		return im.transpose(1,2,0) / 2 + .5

	# def get_example(self, i):
	# 	*im_parts, lab = super(Dataset, self).get_example(i)

	# 	if len(im_parts) == 1:
	# 		return im_parts[0], lab

	# 	else:
	# 		return im_parts[0], im_parts[1], lab

