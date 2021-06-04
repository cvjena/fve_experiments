import chainer

from contextlib import contextmanager
from functools import wraps

@contextmanager
def eval_mode():
	with chainer.no_backprop_mode(), chainer.using_config("train", False):
		yield

def _entropy(var, normalize=True):
	xp = chainer.backend.get_array_module(var)
	array = getattr(var, "array", var)
	mask = array > 0
	ent = -(array[mask] * xp.log(array[mask])).sum()

	if normalize and len(array) not in (1, 0):
		# divide by maximum possible entropy
		ent /= -xp.log(1 / len(array))

	return ent

def _unpack(var):
	return var[0] if isinstance(var, tuple) else var

def tuple_return(method):

	@wraps(method)
	def inner(self, *args, **kwargs):
		res = method(self, *args, **kwargs)
		if not isinstance(res, tuple):
			res = res,
		return res

	return inner
