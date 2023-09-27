import numpy as np
import typing

from chainer import functions as F
from chainer import training as T
from chainer.iterators import SerialIterator
from chainer.optimizer_hooks import WeightDecay
from chainer.optimizers import Adam

from gen_data import data as data_module
from gen_data.classifier.base import Classifier


def concat_var_examples(batch, device=None):

	Xs, ys = zip(*batch)

	Xs = F.stack(Xs, axis=0)
	ys = np.hstack(ys)
	return Xs, ys


def zero_grads(stop_trigger: typing.Tuple[int, str]):
	enabled = stop_trigger[1] == "epoch"

	@T.extension.make_extension(trigger=(1, "epoch"))
	def inner(trainer: T.Trainer):
		if not enabled:
			return

		updater = trainer.updater
		if updater.epoch == stop_trigger[0]:
			return

		data = updater.get_iterator("main").dataset

		data.X.cleargrad()

	return inner

def train(data: data_module.Data, clf: Classifier, *,
		  eval_data: data_module.Data = None,
		  batch_size: int,
		  learning_rate: float,
		  triggers: dict,
		  device: int = -1,
		  decay: float = -1,

		  pre_training_callback = None,
	  ):

	it = SerialIterator(data, batch_size=min(batch_size, len(data)))

	optimizer = Adam(alpha=learning_rate)
	optimizer.setup(clf)

	if decay > 0:
		optimizer.add_hook(WeightDecay(decay))

	updater = T.updaters.StandardUpdater(it, optimizer, converter=concat_var_examples, device=device)


	trainer = T.Trainer(updater=updater, stop_trigger=triggers["stop"], out="/tmp/chainer_logs")

	print_values= ["epoch", "main/accu", "main/loss", "main/dist"]

	if eval_data is not None:
		eval_it = SerialIterator(eval_data, batch_size=batch_size,
												   repeat=False, shuffle=False)
		evaluator = T.extensions.Evaluator(eval_it, target=clf, device=updater.device)
		eval_name = "val"
		evaluator.default_name = eval_name
		trainer.extend(evaluator, trigger=triggers["log"])
		print_values.extend([f"{eval_name}/main/accu", f"{eval_name}/main/loss", f"{eval_name}/main/dist"])


	trainer.extend(T.extensions.LogReport(trigger=triggers["log"]))
	trainer.extend(T.extensions.PrintReport(print_values), trigger=triggers["log"])
	# trainer.extend(T.extensions.ProgressBar(update_interval=triggers["progress_bar"]))
	# trainer.extend(JupyterProgressBar(update_interval=triggers["progress_bar"]))

	trainer.extend(zero_grads(triggers["stop"]))

	if pre_training_callback is not None and callable(pre_training_callback):
		pre_training_callback(trainer)
	trainer.run()
