import chainer

from chainer import functions as F
from chainer import links as L
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from gen_data.utils import plotting

class Classifier(chainer.Chain):

	@classmethod
	def new(cls, opts, *a, **kw):
		args, kwargs = cls.extract_args(opts, *a, **kw)
		return cls(*args, **kwargs)

	@classmethod
	def extract_args(cls, opts, *a, **kw):
		args = tuple(a)
		kwargs = dict(kw,
			n_classes=opts.n_classes,
			in_size=opts.n_dims,
		)
		return args, kwargs

	def report(self, **values):
		chainer.report(values, self)

	def __init__(self, n_classes, in_size=2):
		super(Classifier, self).__init__()
		self.n_classes = n_classes
		with self.init_scope():
			self.fc = L.Linear(in_size=in_size, out_size=n_classes)

	def decision_function(self, X):
		features = self.encode(X)
		return self.classify(features)

	def encode(self, X):
		return X

	def classify(self, features):
		#return self.fc(F.dropout(features))
		return self.fc(features)

	def forward(self, X, y=None):
		logits = self.decision_function(X)

		accu, loss = F.accuracy(logits, y), F.softmax_cross_entropy(logits, y)

		self.report(accu=accu, loss=loss)

		# uncomment this to plot gradients of the FVE w.r.t. the inputs only
		#return F.sum(features)

		# returning the classification loss, plots the gradients of the classification
		# w.r.t. the inputs (composed of the FVE gradient and classifier gradient)
		return loss

	def plot(self, ax: plt.Axes = None, cm: colors.ListedColormap = None):
		ax = ax or plt.gca()
		cm = cm or plt.cm.viridis

		xlim, ylim = ax.get_xlim(), ax.get_ylim()
		xs = np.array(xlim)

		tri = np.tri(self.n_classes, self.n_classes, -1, dtype=np.bool)
		i0s, i1s = np.where(tri)

		W, bias = self.fc.W.array, self.fc.b.array
		for i0, i1 in zip(i0s, i1s):
			w0x, w0y = W[i0]
			b0 = bias[i0]

			w1x, w1y = W[i1]
			b1 = bias[i1]

			a, b  = -(w0x-w1x) / (w0y-w1y), -(b0-b1) / (w0y-w1y)

			ys = a * xs + b

			ax.plot(xs, ys, c="k")


		ax.set_xlim(xlim)
		ax.set_ylim(ylim)

		return ax


def baseline(data, eval_data=None, clf_class=LinearSVC, no_plot=False):

	X, y = data.X.array, data.y
	# scaler = preprocessing.StandardScaler().fit(X)
	# X = scaler.transform(X)
	X_val, y_val = None, None

	if eval_data is not None:
		X_val, y_val = eval_data.X, eval_data.y
		# X_val = scaler.transform(X_val)

	use_dual = X.shape[0] < X.shape[1]
	svm = clf_class(dual=use_dual)

	searcher = GridSearchCV(svm,
		n_jobs=4, param_grid=dict(
			C=[1e-2, 1e-1, 1e0, 1e1, 1e2],
			tol=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
		)).fit(X, y)

	kwargs = searcher.best_params_
	svm = clf_class(dual=use_dual, **kwargs).fit(X, y)

	print(f"Baseline ({clf_class.__name__}) with best params ({kwargs})")
	accu = svm.score(X, y)
	print(f"Training accu:   {accu: 12.4%}")
	if eval_data is not None:
		val_accu = svm.score(X_val, y_val)
		print(f"Validation accu: {val_accu: 12.4%}")

	if X.shape[1] == 2 and not no_plot:
		fig, ax = plt.subplots(figsize=(12,12))
		plotting._plot_decisions(X, y, clf=svm, alpha=0.5, ax=ax)

		# plot the training points
		ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', alpha=0.7)

		if eval_data is not None:
			ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, marker="x", edgecolors='k', alpha=0.7)

		ax.set_xticks(())
		ax.set_yticks(())

	return svm, dict(accu=accu, val_accu=val_accu)
