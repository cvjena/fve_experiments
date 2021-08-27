import numpy as np
from skimage.util.shape import view_as_windows

from fve_fgvc.core.dataset.tmnist import TranslatedMNIST

class ClutteredTranslatedMNIST(TranslatedMNIST):

	@classmethod
	def new(cls, opts, *args, **kwargs):
		_kwargs = dict(
			n_patches=opts.n_patches,
			patch_size=opts.patch_size
		)
		_kwargs.update(kwargs)

		return super(ClutteredTranslatedMNIST, cls).new(opts, *args, **_kwargs)


	def __init__(self, *args, n_patches=10, patch_size=10, **kwargs):
		super(ClutteredTranslatedMNIST, self).__init__(*args, **kwargs)
		self.n_patches = n_patches
		self.patch_size = patch_size

	def new_clutter_index(self):
		return self.rnd.choice(len(self), size=self.n_patches, replace=False)

	def clutter(self, output):
		idxs = self.new_clutter_index()

		_n = np.arange(self.n_patches)
		ims = self._ims[idxs]
		b, c, h, w = ims.shape

		_rnd_pos = lambda max_size: self.rnd.randint(low=0, high=max_size - self.patch_size, size=self.n_patches)

		if self.patch_size < h:
			# 1. extract random patch from the images
			# based on the solution from here: https://stackoverflow.com/a/48191338/1360842
			xs0, ys0 = [_rnd_pos(max_size=s) for s in [w,h]]
			view = view_as_windows(ims, (1, c, self.patch_size, self.patch_size))
			patches = view[_n, 0, ys0, xs0, 0]

		else:
			# 1. or take the entire image as patch
			patches = ims


		# 2. paste it into a random position
		xs0, ys0 = [_rnd_pos(max_size=self.size) for _ in range(2)]
		view = view_as_windows(output, (c, self.patch_size, self.patch_size))
		view[:, ys0, xs0] = patches
		return output

	def transform(self, im, output=None):

		c, h, w = im.shape
		output = self.get_output(output, c=c, dtype=im.dtype)
		output = self.clutter(output)

		return super(ClutteredTranslatedMNIST, self).transform(im, output)
