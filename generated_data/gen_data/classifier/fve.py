import chainer


from chainer import functions as F
from chainer import links as L
from matplotlib import colors
from matplotlib import pyplot as plt

from fve_layer.backends.chainer.links import FVELayer
from fve_layer.backends.chainer.links import FVELayer_noEM
from fve_layer.common.visualization import draw_ellipse

from gen_data import utils
from gen_data.classifier.base import Classifier

class FVEClassifier(Classifier):

	@classmethod
	def extract_args(cls, opts, *a, **kw):
		args, kwargs = super(FVEClassifier, cls).extract_args(opts, *a, **kw)

		n_comps = opts.n_components if opts.n_components > 0 else opts.n_classes
		kwargs = dict(kwargs,
					  in_size=opts.n_dims,
					  embed=opts.embed,
					  n_components=n_comps,
					  normalize=opts.fve_normalize,
					  only_mu=opts.fve_only_mu,
					  only_sig=opts.fve_only_sig,
					  linear_clf=opts.fve_linear,
					  no_update=opts.fve_no_update,
					 )
		return args, kwargs


	def __init__(self, n_classes, in_size=2, *,
				 embed: bool = False,
				 n_components=1, fve_class=FVELayer,
				 linear_clf=False,
				 normalize=False,
				 only_mu=False,
				 only_sig=False,
				 no_update=False,
				 **kwargs):

		factor = 1 if (only_mu or only_sig) else 2
		encoding_size = factor * in_size * n_components

		super(FVEClassifier, self).__init__(n_classes, in_size=encoding_size)

		with self.init_scope():
			if embed:
				embed_init = self.xp.eye(in_size)
				# embedding = chainer.Sequential(
				# 	L.Linear(in_size, in_size, nobias=False),
				# 	F.leaky_relu,
				# 	L.Linear(in_size, in_size, nobias=False),
				# )
				embedding = L.Linear(
					in_size=in_size,
					out_size=in_size,
					initialW=embed_init,
					nobias=False)
			else:
				embedding = None

			self.embedding = embedding
			self.fve_layer = fve_class(in_size=in_size, n_components=n_components, **kwargs)

			if linear_clf:
				post_fve = None
			else:
				post_fve = L.Linear(in_size=encoding_size, out_size=encoding_size)

			self.post_fve = post_fve
			self.add_persistent("is_linear", linear_clf)
			self.add_persistent("normalize", normalize)
			self.add_persistent("only_mu", only_mu)
			self.add_persistent("only_sig", only_sig)
			self.add_persistent("no_update", no_update)

	def distance(self, x):
		dist = self.fve_layer.mahalanobis_dist(x)
		assingment = self.fve_layer.soft_assignment(x)
		return F.mean(F.sum(assingment * dist, axis=-1))

	def encode(self, X):
		if self.embedding is not None:
			X = self.embedding(X)

		x = F.expand_dims(X, axis=1)

		if self.no_update:
			self.fve_layer.disable_update()
			with chainer.using_config("train", False):#, chainer.no_backprop_mode():
				enc = self.fve_layer(x)

		else:
			enc = self.fve_layer(x)


		self.report(dist=self.distance(x))


		if self.normalize:
			enc = F.normalize(enc)

		_, size = enc.shape

		if self.only_mu and self.only_sig:
			raise ValueError("Setting both only_mu and only_sig is not allowed!")

		elif self.only_mu and not self.only_sig:
			enc = enc[:, :size//2]

		elif not self.only_mu and self.only_sig:
			enc = enc[:, size//2:]

		return enc

	def classify(self, features):
		if self.post_fve is not None:
			features = F.relu(self.post_fve(features))

		elif not self.is_linear:
			features = F.relu(features)

		return super(FVEClassifier, self).classify(features)


	def plot(self, ax: plt.Axes = None, cm: colors.ListedColormap = None):
		if self.fve_layer.in_size != 2: return

		mu = utils.get_array(self.fve_layer.mu)
		sig = utils.get_array(self.fve_layer.sig)

		for _mu, _sig in zip(mu.T, sig.T):
			ax.scatter(*_mu, marker="x", color="black")
			draw_ellipse(_mu, _sig,
				nsig=2, ax=ax,
				alpha=0.7,
				facecolor="none",
				edgecolor="black", lw=3,
			)


		return ax
