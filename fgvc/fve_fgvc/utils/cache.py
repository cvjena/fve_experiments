import functools
import logging
import multiprocessing as mp
import pathlib
import pickle
import tempfile
import warnings
import uuid


class Cache:


	def __init__(self, *args, cache_folder=None, **kwargs):
		super().__init__(*args, **kwargs)

		warnings.warn("This cache only makes sence on really fast devices!")

		self._manager = mp.Manager()
		self._manager.__enter__()
		self.cache = self._manager.dict()

		if cache_folder is None:
			logging.info(f"No TempDir is created. Objects are cached in memory.")
			self.directory = None
		else:
			self.directory = tempfile.TemporaryDirectory(dir=cache_folder)
			logging.info(f"Creating TempDir context: {self.directory}...")
			self.directory.__enter__()


	@property
	def root(self):
		if self.has_directory:
			return pathlib.Path(self.directory.name)

	@property
	def has_directory(self):
		return getattr(self, "directory", None) is not None

	def __del__(self):
		print("Closing TempDir context...")
		if self.has_directory:
			self.directory.__exit__(None, None, None)

		if hasattr(self, "_manager"):
			self._manager.__exit__(None, None, None)

	def __contains__(self, key):
		return key in self.cache

	def __getitem__(self, key):
		value = self.cache[key]
		root = self.root

		if root is None:
			return value

		with open(root / value, "rb") as f:
			return pickle.load(f)

	def __setitem__(self, key, value):

		root = self.root
		if root is None:
			self.cache[key] = value
			return

		self.cache[key] = fname = str(uuid.uuid4())
		with open(self.root / fname, "wb") as f:
			pickle.dump(value, f)

	@staticmethod
	def default_getter(_self, *args, **kwargs):
		return args[0]

	@staticmethod
	def cached(func=None, *, key_getter=None):
		if key_getter is None:
			key_getter = Cache.default_getter

		assert func is None or callable(func), "Wrong usage of the decorator!"
		assert callable(key_getter)

		def _deco(func):
			@functools.wraps(func)
			def wrapper(self, *args, **kwargs):

				key = key_getter(self, *args, **kwargs)

				if self._cache is not None and key in self._cache:
					return self._cache[key]

				value = func(self, *args, **kwargs)

				if self._cache is not None:
					self._cache[key] = value

				return value

			return wrapper

		return _deco(func) if callable(func) else _deco






if __name__ == '__main__':

	class Foo:
		_cache = None

		@Cache.cached()
		def bar(self, obj):
			print("<bar>", self, obj)

		@Cache.cached
		def bar2(self, obj):
			print("<bar2>", self, obj)

	print("Object created")
	foo = Foo()
	print("Calling method bar")
	foo.bar(2)
	print("Calling method bar2")
	foo.bar2(2)
