import chainer
import logging

from fve_example.core.dataset.mnist import MNIST
from fve_example.core.dataset.tmnist import TranslatedMNIST
from fve_example.core.dataset.ctmnist import ClutteredTranslatedMNIST

from cvdatasets.utils import new_iterator


DATASETS = dict(
	mnist=MNIST,
	tmnist=TranslatedMNIST,
	ctmnist=ClutteredTranslatedMNIST,
)

def new_datasets(args):

	assert args.dataset in DATASETS
	ds_cls = DATASETS[args.dataset]

	train, test = [ds_cls.new(args, raw_data=data) for data in chainer.datasets.get_mnist(ndim=3)]

	logging.info(f"Created {ds_cls.__name__} datasets with {len(train)} train and {len(test)} test images")

	return train, test

def new_iterators(args, train, test):
	it_kwargs = dict(
		n_jobs=args.n_jobs,
		batch_size=args.batch_size,
	)

	train_it, _ = new_iterator(train, **it_kwargs)
	test_it, _ = new_iterator(test,
		repeat=False, shuffle=False, **it_kwargs)

	return train_it, test_it
