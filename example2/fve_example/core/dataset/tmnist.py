import numpy as np

from fve_example.core.dataset.mnist import MNIST

class TranslatedMNIST(MNIST):

	def __init__(self, *args, seed=None, size=64, **kwargs):
	    super(TranslatedMNIST, self).__init__(*args, **kwargs)

	    self.rnd = np.random.RandomState(seed)
	    self.size = size

	def get_example(self, i):
		im, lab = super(TranslatedMNIST, self).get_example(i)

		new_im = self.transform(im)

		return new_im, lab

	def get_output(self, output, c=3, dtype=np.float32):
		new_shape = (c, self.size, self.size)

		if output is None:
			output = np.zeros(new_shape, dtype=dtype)
		else:
			assert output.shape == new_shape, \
				f"shape of the given image does not match: {output.shape} != {shape}"

		return output

	def transform(self, im, output=None):
		c, h, w = im.shape
		new_im  = self.get_output(output, c=c, dtype=im.dtype)

		x0, y0 = [self.rnd.randint(0, self.size-s) for s in [w, h]]
		new_im[..., y0:y0+h, x0:x0+w] = im

		return new_im
