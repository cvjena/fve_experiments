import logging
import pickle
import tempfile
import uuid
import pathlib
import multiprocessing as mp

class ImageCache:

	def __init__(self, *args, cache_folder, **kwargs):
		super().__init__(*args, **kwargs)
		# comment this out, if you want to use this implementation
		raise RuntimeError("this cache only makes sence on really fast devices!")

		self._manager = mp.Manager()
		self._manager.__enter__()
		self.cache = self._manager.dict()

		self.directory = tempfile.TemporaryDirectory(dir=cache_folder)
		logging.info(f"Creating TempDir context: {self.directory}...")

		self.directory.__enter__()

	@property
	def root(self):
		return pathlib.Path(self.directory.name)

	def __del__(self):
		print("Closing TempDir context...")
		self.directory.__exit__(None, None, None)
		self._manager.__exit__(None, None, None)

	def __contains__(self, key):
		return key in self.cache

	def __getitem__(self, key):

		fname = self.cache[key]
		with open(self.root / fname, "rb") as f:
			return pickle.load(f)

	def __setitem__(self, key, value):

		fname = str(uuid.uuid4())
		with open(self.root / fname, "wb") as f:
			pickle.dump(value, f)
		self.cache[key] = fname
