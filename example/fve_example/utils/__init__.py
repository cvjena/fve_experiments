import chainer

from contextlib import contextmanager
from functools import wraps

@contextmanager
def eval_mode():
	with chainer.no_backprop_mode(), chainer.using_config("train", False):
		yield

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
