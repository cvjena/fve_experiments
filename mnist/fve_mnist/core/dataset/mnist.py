import numpy as np

from chainer.dataset import DatasetMixin
from cvdatasets.dataset import ImageProfilerMixin

class MNIST(DatasetMixin, ImageProfilerMixin):

	@classmethod
	def new(cls, opts, *args, **kwargs):
		return cls(*args, **kwargs)

	def __init__(self, raw_data, *args, **kwargs):
		super(MNIST, self).__init__()
		self._raw_data = raw_data
		self._ims, self._labels = map(np.array, zip(*raw_data))

		# N x 1 x 28 x 28 -> N x 1 x 32 x 32
		self._ims = np.pad(self._ims, ((0,0), (0,0), (2,2), (2,2)))

		# N x 1 x 32 x 32 -> N x 3 x 32 x 32
		self._ims = np.repeat(self._ims, 3, axis=1)

	def get_example(self, i):
		return self._ims[i], self._labels[i]

	def __len__(self):
		return len(self._labels)
