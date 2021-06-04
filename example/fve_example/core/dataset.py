import chainer
import logging
import numpy as np

from chainercv import transforms as tr
from functools import wraps

from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import ImageProfilerMixin
from cvdatasets.dataset import TransformMixin
from cvdatasets.dataset import UniformPartMixin
from cvdatasets.utils import transforms as tr2


def get_params(opts) -> dict:

	return dict(
		dataset_cls=Dataset,
		dataset_kwargs_factory=Dataset.kwargs
	)

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

class Dataset(ImageProfilerMixin, TransformMixin, UniformPartMixin, AnnotationsReadMixin):
	label_shift = None

	@classmethod
	def kwargs(cls, opts, subset) -> dict:
		return dict(opts=opts)

	def __init__(self, *args, opts, prepare, center_crop_on_val, **kwargs):
		super(Dataset, self).__init__(*args, **kwargs)
		self.prepare = prepare
		self.center_crop_on_val = center_crop_on_val
		self._cache = None #{} if opts.cache_images else None

		# for these models, we need to scale from 0..1 to -1..1
		self.zero_mean = opts.model_type in ["cvmodelz.InceptionV3"]
		self.shuffle_parts = opts.shuffle_parts
		if self.shuffle_parts:
			logging.info("=== Order of the parts will be shuffled! ===")

		self._setup_augmentations(opts)

		if opts.only_klass is not None:
			mask = self.labels < opts.only_klass
			self._orig_uuids = self.uuids
			self.uuids = self.uuids[mask]

	def _setup_augmentations(self, opts):

		min_value, max_value = (0, 1) if self.zero_mean else (None, None)

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

		if self.center_crop_on_val:
			logging.info("During evaluation, center crop is used!")
			self._val_augs.append(pos_augs["center_crop"])

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

		if self.shuffle_parts:
			np.random.shuffle(parts)

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
					params = dict(params, size=self._part_size)

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

		im = im.astype(chainer.config.dtype)
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

