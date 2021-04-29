import chainer

from contextlib import contextmanager

@contextmanager
def eval_mode():
	with chainer.no_backprop_mode(), chainer.using_config("train", False):
		yield
