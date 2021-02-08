import numpy as np
import datetime
import chainer

from os.path import join

from chainer import reporter as reporter_module
from chainer.backends import cuda
from chainer.dataset import convert
from chainer.training import extension
from chainer.training.extensions import util



class FeatureStatistics(extension.Extension):

	trigger = 1, "epoch"
	default_name = "feat_stats"
	priority = extension.PRIORITY_WRITER

	# will be set to default_name by trainer
	name = None

	def __init__(self, target: chainer.Chain, train_it,
				 val_it=None,
				 converter=convert.concat_examples,
				 device=None, **kwargs):

		self.target = target
		self.train_it = train_it
		self.val_it = val_it

		self.converter = converter
		self.device = device

	def analyze(self, *args, **kwargs):
		try:
			reporter = reporter_module.get_current_reporter()
		except Exception as e:
			reporter = reporter_module.Reporter()

		with reporter.scope({}):
			return self._analyze(*args, **kwargs)

	def _analyze(self, it, report_prefix=""):
		if it is None:
			return

		clf = self.target

		mu0, var0 = [cuda.to_cpu(getattr(p, "array", p)[:, 0])
			for p in [clf.fve_layer.mu, clf.fve_layer.sig]]


		it.reset()
		convs = None
		n_samples = len(it.dataset)
		n_batches = int(np.ceil(n_samples / it.batch_size))

		with ProgressBar(iterator=it) as pbar:
			for i in np.arange(n_batches):
				batch = it.next()
				inputs = convert._call_converter(
					self.converter, batch, self.device)
				if len(inputs) == 2:
					return

				if len(inputs) == 3:
					X, parts, y = inputs

				else:
					raise ValueError("Input ill-formed!")

				# conv_map0 = clf._get_conv_map(X, model=clf.separate_model)
				# conv_map1 = clf._call_pre_fve(conv_map0)

				if parts is None:
					continue
				n, t, c, h, w = parts.shape
				_parts = parts.reshape(n*t, c, h, w)
				part_convs0 = clf._get_conv_map(_parts, model=clf.model)
				part_convs1 = clf._call_pre_fve(part_convs0)

				_, *rest = part_convs1.shape

				if convs is None:
					convs = np.zeros((n_samples, t,) + tuple(rest), dtype=np.float32)

				n0 = int(i * it.batch_size)
				n1 = n0 + n
				offset = n1 - n_samples if n1 > n_samples else 0
				_part_convs = cuda.to_cpu(part_convs1.array.reshape(n, t, *rest))
				convs[n0: n1-offset] = _part_convs[:len(_part_convs)-offset]

				pbar.update()

		if convs is None:
			return None, None

		mu, var = convs.mean(axis=(0,1,3,4)), convs.var(axis=(0,1,3,4))

		return convs, {
			# mu = \u03BC
			f"{report_prefix}main/mu/min": np.min(mu),
			f"{report_prefix}main/mu/max": np.max(mu),
			# sigma = \u03C3
			f"{report_prefix}main/sig/min": np.min(var),
			f"{report_prefix}main/sig/max": np.max(var),

			# mu0
			f"{report_prefix}main/mu0/min": np.min(mu0),
			f"{report_prefix}main/mu0/max": np.max(mu0),
			# sigma0
			f"{report_prefix}main/sig0/min": np.min(var0),
			f"{report_prefix}main/sig0/max": np.max(var0),

			# mean((mu0 - mu)**2)
			f"{report_prefix}main/mu/mse": np.mean((mu - mu0)**2),
			# mean((sigma0 - sigma)**2)
			f"{report_prefix}main/sig/mse": np.mean((var - var0)**2),

		}

	def __call__(self, trainer=None, conv_dump=None):

		with chainer.using_config("train", False), chainer.no_backprop_mode():
			train_convs, train_stats = self.analyze(self.train_it)
			val_convs, val_stats = self.analyze(self.val_it, report_prefix="val/")

		chainer.report(train_stats)

		if conv_dump is not None:
			np.savez(join(conv_dump, "train_convs.npz"), train_convs)

		if val_stats is not None:
			chainer.report(val_stats)

			if conv_dump is not None:
				np.savez(join(conv_dump, "val_convs.npz"), val_convs)


		return dict(train_stats=train_stats, val_stats=val_stats)



	def initialize(self, trainer):
		super(FeatureStatistics, self).initialize(trainer)

	def finalize(self):
		super(FeatureStatistics, self).finalize()


## copied from chainer.training.extensions.evaluator
class ProgressBar(util.ProgressBar):

	def __init__(self, iterator, bar_length=None, out=None):
		if not (hasattr(iterator, 'current_position') and
				hasattr(iterator, 'epoch_detail')):
			raise TypeError('Iterator must have the following attributes '
							'to enable a progress bar: '
							'current_position, epoch_detail')
		self._iterator = iterator

		super(ProgressBar, self).__init__(
			bar_length=bar_length, out=out)

	def get_lines(self):
		iteration = self._iterator.current_position
		epoch_detail = self._iterator.epoch_detail
		epoch_size = getattr(self._iterator, '_epoch_size', None)

		lines = []

		rate = epoch_detail
		marks = '#' * int(rate * self._bar_length)
		lines.append('feature stats [{}{}] {:6.2%}\n'.format(
					 marks, '.' * (self._bar_length - len(marks)), rate))

		if epoch_size:
			lines.append('{:10} / {} iterations\n'
						 .format(iteration, epoch_size))
		else:
			lines.append('{:10} iterations\n'.format(iteration))

		speed_t, speed_e = self.update_speed(iteration, epoch_detail)
		estimated_time = (1.0 - epoch_detail) / speed_e
		lines.append('{:10.5g} iters/sec. Estimated time to finish: {}.\n'
					 .format(speed_t,
							 datetime.timedelta(seconds=estimated_time)))
		return lines

	def __enter__(self):
		return self

	def __exit__(self, *args, **kwargs):
		self.close()
