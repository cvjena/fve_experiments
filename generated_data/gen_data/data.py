import chainer
import numpy as np

from chainer.dataset import DatasetMixin
from matplotlib import colors
from matplotlib import pyplot as plt
from typing import Callable

from gen_data import utils

def _new_data(n_samples, n_dims, mean, std, rnd=None):
	""" generates normal distributed data """
	rnd = rnd or np.random.RandomState()
	return rnd.normal(loc=mean, scale=std, size=(n_samples, n_dims))

def _new_data2(n_samples, n_dims, mean, std, rnd=None, *, n_circles=4):
	""" generates data as regular circles """
	assert n_dims == 2, "Only 2-D data is supported"

	factors = np.linspace(0.5, 2, num=n_circles) # eg. default is [0.5, 1.0, 1.5, 2.0]
	# in each row is one of the factors multiplied with the std
	_stds = np.outer(factors, std)
	_n_samples = n_samples // n_circles

	_X = np.zeros((n_samples, n_dims), dtype=mean.dtype)

	for i, _std in enumerate(_stds):
		# distribute the samples in a circle
		angles = np.linspace(0, 2*np.pi, _n_samples, endpoint=False)
		xs = np.vstack([np.cos(angles), np.sin(angles)]).T

		i0 = i * _n_samples
		i1 = i0 + _n_samples
		_X[i0:i1] = xs * _std + mean

	return _X

def _new_rnd(seed, is_evaluation):
	if seed is not None and is_evaluation:
		seed = 2**31 - seed

	return np.random.RandomState(seed)

class Data(DatasetMixin):
	@classmethod
	def new(cls, args, evaluation: bool=False):
		return cls(
			n_classes=args.n_classes,
			n_dims=args.n_dims,
			n_samples=args.n_samples,
			std=args.sample_std,
			scale=args.data_scale,
			shift=args.data_shift,
			rnd=_new_rnd(args.seed, evaluation)
		)

	def __init__(self, *, n_classes, n_dims, n_samples, std=1, dtype=np.float32, scale=None, shift=None, rnd=None):
		super(Data, self).__init__()

		self.n_classes = n_classes
		self.n_samples = n_samples
		self.n_dims = n_dims

		self.X = np.zeros((n_classes * n_samples, n_dims), dtype=dtype)
		self.y = np.zeros(n_classes * n_samples, dtype=np.int32)

		# distribute the class centers evenly in a circle (in the 1st two dimensions)
		_comp_pos = np.linspace(0, 2*np.pi, n_classes, endpoint=False)
		self._means = np.zeros((n_classes, n_dims), dtype=np.float32)
		self._means[:, :2] = np.vstack([np.cos(_comp_pos), np.sin(_comp_pos)]).T

		self._std = np.ones(n_dims, dtype=np.float32) / n_classes * std


		for i, mean in enumerate(self._means):
			_X = _new_data(n_samples, n_dims, mean, self._std, rnd)
#             _X = _new_data2(n_samples, mean, self._std, rnd)
			n0 = i * self.n_samples
			n1 = n0 + self.n_samples
			self.X[n0: n1] = _X
			self.y[n0: n1] = i

		if scale is not None:
			self.X *= scale
			self._means *= scale
			self._std *= scale

		if shift is not None:
			self.X += shift
			self._means += shift

		self.X = chainer.Variable(self.X, name="data")

	def __len__(self):
		return len(self.y)

	def get_example(self, i):
		return self.X[i], self.y[i]

	def plot(self,
			 ax: plt.Axes = None,
			 cm: colors.ListedColormap = None,
			 marker: str = None,
			 alpha: float = 1.0,
			 plot_grad=False,
			 embedding: Callable = None,
			 **kwargs,
			):

		ax = ax or plt.gca()
		cm = cm or plt.cm.viridis
		_x = self.X if embedding is None else embedding(self.X)
		_x = chainer.cuda.to_cpu(utils.get_array(_x))
		_y = chainer.cuda.to_cpu(self.y)
		ax.scatter(*_x.T, c=cm(_y / self.n_classes), marker=marker, alpha=alpha)
		if plot_grad and self.X.grad is not None:
			self._plot_grads(ax, cm, **kwargs)
		return ax


	def _plot_grads(self,
					ax: plt.Axes,
					cm: colors.ListedColormap,
					norm: bool = True,
					xp=np):

		x0 = self.X.array
		step_size = 1e-1

		grad = self.X.grad.copy()
		if norm:
			grad_norm = xp.sqrt(xp.sum(grad **2, axis=1))
			mask = grad_norm != 0
			grad[mask] = grad[mask] / grad_norm[mask, None]

		for (x, y), (dx, dy), lab in zip(x0, -grad, self.y):
			if dx == dy == 0:
				continue
			ax.arrow(x,y, dx, dy, width=2e-1, facecolor=cm(lab / self.n_classes))

		return ax
