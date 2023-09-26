import logging

from chainer.training import extensions
from cvfinetune.training.trainer import Trainer as DefaultTrainer

# import numpy as np
# import pyaml
# import gc

# from bdb import BdbQuit
# from pathlib import Path
# from tqdm.auto import tqdm

# from chainer.backends import cuda
# from chainer.dataset import convert
# from chainer.serializers import save_npz
# from chainer.training.extension import make_extension
# from chainer_addons.training import lr_shift
# from chainer_addons.training import optimizer
# from cvdatasets.utils import attr_dict
# from cvdatasets.utils import new_iterator
# from cvfinetune.finetuner import DefaultFinetuner
# from cvfinetune.training.trainer.base import default_intervals
# from fve_fgvc.core.training.extensions import FeatureStatistics
# from fve_fgvc.core.training.updater import updater_params

def trainer_params() -> dict:
	return dict(trainer_cls=Trainer)

class Trainer(DefaultTrainer):

	def __init__(self, opts, *args, **kwargs):
		super().__init__(opts=opts, *args, **kwargs)

		ext = extensions.ExponentialShift(
			attr="aux_lambda",
			init=opts.aux_lambda,
			rate=opts.aux_lambda_rate,
			optimizer=self.clf)

		self.extend(ext, trigger=(opts.aux_lambda_step, 'epoch'))
		logging.info(f"Aux impact starts at {opts.aux_lambda} "
			f"and is reduced by {opts.aux_lambda_rate} "
			f"every {opts.aux_lambda_step} epoch")


	def reportables(self, args):

		print_values = [
			"elapsed_time",
			"epoch",
			# "lr",

			self.eval_name("main/gamma"),
			"main/accu", self.eval_name("main/accu"),
			"main/loss", self.eval_name("main/loss"),

		]

		plot_values = {
			"accuracy": [
				"main/accu",  self.eval_name("main/accu"),
			],
			"loss": [
				"main/loss", self.eval_name("main/loss"),
			],
		}

		# if args.fve_type != "no":

		# 	print_values.extend([
		# 		"main/w_ent", self.eval_name("main/w_ent"),
		# 	])

		if args.parts != "GLOBAL":
			print_values.extend([
				"main/g_accu", self.eval_name("main/g_accu"),
			])

			print_values.extend([
				"main/p_accu", self.eval_name("main/p_accu"),
			])

			if args.aux_lambda > 0:

				print_values.extend([
					"main/aux_accu", self.eval_name("main/aux_accu"),
					"main/aux_p_accu", self.eval_name("main/aux_p_accu"),
				])

		return print_values, plot_values
