import chainer
import logging
import numpy as np


from chainer.backends.cuda import get_device
from chainer.backends.cuda import to_cpu
from chainer.dataset import convert

from cvdatasets.utils import new_iterator
from matplotlib import pyplot as plt
from tqdm import tqdm



class Visualizer(object):

	@classmethod
	def new(cls, args, target, dataset, **kwargs):
		assert args.parts not in ["GLOBAL"], \
			f"invalid parts selected: {args.parts}"

		classes = args.classes

		if classes:
			logging.info(f"Selected {len(classes)} classes!")
			mask = np.in1d(dataset.labels, classes)
			prev_len = len(dataset)
			dataset.uuids = dataset.uuids[mask]
			logging.info(f"Reduced # of samples from {prev_len} to {len(dataset)}")

		else:
			logging.info("No classes were selected, visualizing all classes!")

		return cls(args, target, dataset, **kwargs)


	def __init__(self, opts, target, dataset, **kwargs):
		super(Visualizer, self).__init__()

		self.target = target
		self.it, self._n_batches = new_iterator(dataset,
			n_jobs=opts.n_jobs,
			batch_size=opts.batch_size,
			repeat=False,
			shuffle=False)

		self.device = opts.gpu[0]

		if self.device >= 0:
			get_device(self.device).use()
			target.to_device(self.device)

		self._load_class_names(opts.class_names)

	def _load_class_names(self, names_file):
		if names_file is None:
			self.class_names = None
			return

		names = np.loadtxt(names_file, dtype=str)
		self.class_names = {int(cls_id): cls_name for cls_id, cls_name in names}
		logging.info(f"Class names are loaded from \"{names_file}\"")


	def get_features(self):
		self.it.reset()

		features = []
		for i, batch in tqdm(enumerate(self.it), total=self._n_batches):
			ims, parts, y = convert.concat_examples(batch, device=self.device)
			feats = self.target.get_part_features(parts)

			features.extend(to_cpu(feats.array))

		return np.array(features)


	def get_distances(self, features, comp_selection="weighted"):
		"""
			Component selection options:
				- weighted: weighted sum according to the soft assignments
				- max: select the mixture with the greates weight
		"""

		features = _reshape(features)
		fve_layer = self.target.fve_layer

		if fve_layer is None:
			_feat_lens = np.sqrt(np.sum(features ** 2, axis=-1))
			_mean_feat_lens = _feat_lens.mean(axis=1, keepdims=True)
			_feats_to_select = np.where(_feat_lens >= _mean_feat_lens)
			selection = np.zeros(features.shape, dtype=np.float32)
			selection[_feats_to_select]  = 1

			return (features * selection)

		features = chainer.Variable(self.target.xp.array(features))
		mask_idxs = fve_layer.get_mask(features, True)
		mask = self.target.xp.zeros(features.shape, np.float32)
		mask[mask_idxs] = 1

		if comp_selection == "weighted":
			_mu = fve_layer.mu
			assignment = fve_layer.soft_assignment(features).array
			dists = _mu[None, None, :, :] - features[:, :, :, None]

			dists = (dists * assignment[:, :, None, :]).array.sum(axis=-1)

		elif comp_selection == "max":
			max_comp = fve_layer.w.argmax()
			_mu = fve_layer.mu[:, max_comp]
			dists = _mu[None, None, :] - features

		else:
			raise ValueError(f"Unknown comp selection method: {comp_selection}")

		return to_cpu(dists * mask)

	def run(self):
		with chainer.using_config("train", False), chainer.no_backprop_mode():
			feats = self.get_features()

		dists = self.get_distances(feats)

		logging.info(f"converted features ({feats.shape}) to distances ({dists.shape})")
		labs = self.it.dataset.labels[:, None, None]
		labs = np.broadcast_to(labs, dists.shape)

		n, t, c, h, w = feats.shape
		part_assignment = np.broadcast_to(np.arange(t)[:, None, None, None], feats.shape)
		part_assignment = _reshape(part_assignment)

		feat_mask = (dists!=0).max(axis=-1)
		logging.info(f"{feat_mask.mean():.3%} of conv map features are retained")
		idxs = np.where(feat_mask)

		sel_labs = labs[idxs][:, 0]
		sel_dists = dists[idxs]
		sel_parts = part_assignment[idxs]

		from cuml.manifold import TSNE as cuTSNE
		reducer = cuTSNE(2, perplexity=50)
		twoD_dists = reducer.fit_transform(sel_dists)

		grid = plt.GridSpec(2, 4)
		fig = plt.figure(figsize=(18,9))

		ax0 = plt.subplot(grid[:, :2])

		_scatter(ax0, twoD_dists.T, sel_labs, self.class_names,
				 legend=True, alpha=0.4)

		for i, marker in enumerate(["x", "v", ".", "_"]):
			row, col = np.unravel_index(i, (2,2))
			ax = plt.subplot(grid[row, col+2])
			ax.set_title(f"Part #{i+1}")
			marker_mask = sel_parts[:, 0] == i
			xy = twoD_dists[marker_mask].T
			_labs = sel_labs[marker_mask]
			_scatter(ax, twoD_dists.T, sel_labs, self.class_names,
				legend=False, alpha=0.02)
			_scatter(ax, xy, _labs, self.class_names,
				alpha=0.4)

		plt.show()
		plt.close()

		# import pdb; pdb.set_trace()




def _reshape(arr, arr_module=np, xp=np):

	n, t, c, h, w = arr.shape
	# N x T x C x H x W -> N x T x H x W x C
	arr = arr_module.transpose(xp.array(arr), (0, 1, 3, 4, 2))
	# N x T x H x W x C -> N x T*H*W x C
	return arr_module.reshape(arr, (n, t*h*w, c))


def _scatter(ax, xy, labs=None, class_names=None, cmap=plt.cm.inferno, legend=False, **kwargs):

	if labs is None:
		labs = np.zeros(len(xy), dtype=np.int32)

	_labs = np.unique(labs)
	cmap = plt.cm.get_cmap(cmap, len(_labs))
	for i, lab in enumerate(_labs, 0):
		mask = labs==lab
		c = cmap(i / len(_labs))
		label = None if class_names is None else class_names.get(lab)
		ax.scatter(*xy[:, mask], color=c, label=label, **kwargs)

	if legend and class_names is not None:
		leg = ax.legend()
		for lh in leg.legendHandles:
			lh.set_alpha(1)
