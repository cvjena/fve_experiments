import chainer
import numpy as np

from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn.decomposition import PCA

from cluster_parts.core import BoundingBoxPartExtractor
from cluster_parts.core import Corrector
from cluster_parts.utils import ThresholdType
from cluster_parts.utils import FeatureType
from cluster_parts.utils import ClusterInitType
from cluster_parts.utils import operations

_to_cpu = lambda arr: chainer.cuda.to_cpu(chainer.as_array(arr))

def _normalize(arr, chan_axis=2, channel_wise=False):
	arr = _to_cpu(arr)
	return operations.normalize(arr, axis=chan_axis, channel_wise=channel_wise)

def get_extractor(
	gamma=0.7,
	sigma=5.0,
	K=4,
	fit_object=False,
	thresh_type=ThresholdType.MEAN,
	# thresh_type=ThresholdType.PRECLUSTER,
	comp=[
		FeatureType.COORDS,
		FeatureType.SALIENCY,
		# FeatureType.RGB,
		],
	):

	return BoundingBoxPartExtractor(
		corrector=Corrector(gamma=gamma, sigma=sigma),

		K=K,
		optimal=True,
		fit_object=fit_object,

		thresh_type=thresh_type,
		cluster_init=ClusterInitType.MAXIMAS,
		feature_composition=comp,
	)


def _pca(arr, n=2):

	pca = PCA(n_components=n)
	C, H, W = arr.shape
	X = arr.transpose(1, 2, 0).reshape(H*W, C)

	x = pca.fit_transform(X)
	return x.reshape(H, W, n).transpose(2, 0, 1)

class Visualizer:
	def __init__(self, clf, squeeze_saliency: bool = True):
		super().__init__()
		self.clf = clf
		self.squeeze_saliency = squeeze_saliency
		self.cs_parts = get_extractor()

	@property
	def model(self):
		return self.clf.model

	def _prepare_back(self, x):
		model_name = self.model.meta.name
		x = _to_cpu(x).transpose(1,2,0)
		if "resnet" in model_name.lower():
			x += np.array([103.063, 115.903, 123.152], dtype=x.dtype)
			return x[..., ::-1].astype(np.uint8)

		elif "inception" in model_name.lower():
			x = (x + 1) / 2
			return x.astype(np.float32)

		else:
			raise NotImplementedError(f"prepare_back is not implemented for {model_name}!")

	def __call__(self, X):

		cams = self.clf.cam(X, dtype=np.float32)
		with chainer.using_config("train", False), chainer.no_backprop_mode():
			convs, *_ = self.clf.extract(X)

		cpu_convs = _to_cpu(convs).astype(np.float32)
		extractor = self.cs_parts

		for X, conv, cam in zip(X, cpu_convs, cams):
			fig, axs = plt.subplots(2,2, figsize=(16,9), squeeze=False)

			# fig.suptitle(f"Predicted Top-5 Class-IDs: {', '.join(map(str, cls[:5]))}")
			_x = self._prepare_back(X)
			axs[0, 0].imshow(_x)
			axs[0, 0].set_title("Input")
			arrs = [
				(conv, "conv"),
				# (_pca(conv, n=256), "PCA-conv"),
				(None, "nothing"),
				# ((conv + cam) / 2, "conv+cam"),
				# ((np.sqrt(conv * cam)), "conv*cam"),
				# (att, "attention"),
				# (fin_conv, "result"),
				(cam, "cam"),
				# (_pca(cam, n=256), "PCA-cam"),
			]

			size = _x.shape[:-1]
			for i, (arr, title) in enumerate(arrs, 1):
				ax = axs[np.unravel_index(i, axs.shape)]
				if arr is None:
					ax.axis("off")
					continue

				# CxHxW -> HxWxC
				arr = arr.transpose(1,2,0)


				if self.squeeze_saliency:
					l2_arr = operations.l2_norm(arr)
					arr = _normalize(l2_arr)
					norm_l2_arr = arr = resize(arr, size, order=0, mode="edge", preserve_range=True)
				else:
					arr = _normalize(arr)
					arr = resize(arr, size, order=0, mode="edge", preserve_range=True)
					l2_arr = operations.l2_norm(arr)
					norm_l2_arr = _normalize(l2_arr)

				parts = extractor(_x, arr)

				arr = extractor.corrector(arr)
				centers, labs = extractor.cluster_saliency(_x, arr)
				thresh_mask = extractor.thresh_type(_x, arr)
				# parts = extractor.get_boxes(centers, labs, arr)


				if isinstance(thresh_mask, (int, float, arr.dtype.type)):
					thresh_mask = l2_arr > thresh_mask

				norm_l2_arr[~thresh_mask] = 0
				ax.set_title(f"{title} [{(thresh_mask).sum():,d} pixels ({thresh_mask.mean():.2%}) selected]")

				ax.imshow(_x, alpha=1.0)
				ax.imshow(norm_l2_arr, alpha=0.75)
				ax.imshow(labs, alpha=0.4)

				for part_id, ((x,y), w, h) in parts:
					ax.add_patch(plt.Rectangle((x,y), w, h, fill=False))

			plt.show()
			plt.close()

