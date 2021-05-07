import numpy as np

from matplotlib import pyplot as plt

from gen_data import classifier
from gen_data import data as data_module

def _plot_decisions(X: np.ndarray, y: np.ndarray, clf,
	ax=None, resolution=0.01, alpha=0.5):

	if X.shape[1] != 2:
		return

	ax = ax or plt.gca()

	# create the grid for background colors
	x_min, x_max = X[:, 0].min() , X[:, 0].max()
	y_min, y_max = X[:, 1].min() , X[:, 1].max()
	xx, yy = np.meshgrid(
		np.arange(x_min, x_max*(1 + resolution), resolution*(x_max-x_min)),
		np.arange(y_min, y_max*(1 + resolution), resolution*(y_max-y_min))
	)

	# plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, x_max]*[y_min, y_max].
	_X = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
	if hasattr(clf, "decision_function"):
		decision = clf.decision_function(_X)
		Z = getattr(decision, "array", decision).argmax(axis=1)
	else:
		Z = clf.predict_proba(_X)[:, 1]

	Z = Z.reshape(xx.shape)

	# import pdb; pdb.set_trace()
	ax.imshow(Z[::-1],
		alpha=alpha,
		extent=(x_min, x_max, y_min, y_max),
		interpolation="lanczos",
	)
	ax.contour(xx, yy, Z, alpha=1.0)

def _plot_params(data: data_module.Data, clf,
				 clf_dump=None,
				 eval_data: data_module.Data = None,
				 title: str = None,
				 plot_grad: bool = False,
				 plot_norm_grad: bool = False,
				 fig_axs=None):


	if fig_axs is None:
		if clf_dump is None:
			ax0 = None
			fig, ax1 = plt.subplots(figsize=(12,12))

		else:
			fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16,9))
			ax0.set_title("Before Training")
			ax1.set_title("After Training")

	else:
		fig, axs = fig_axs
		try:
			ax0, ax1 = axs
		except:
			ax0, ax1 = None, axs


	if title is not None:
		fig.suptitle(title)

	if ax0 is not None:
		data.plot(ax=ax0)
		if eval_data is not None:
			eval_data.plot(ax=ax0, marker="x", alpha=0.5)
		clf_dump.plot(ax=ax0)

	data.plot(ax=ax1, plot_grad=plot_grad, norm=plot_norm_grad)
	if eval_data is not None:
		eval_data.plot(ax=ax1, marker="x", alpha=0.5)
	clf.plot(ax=ax1)

	return fig, (ax0, ax1)
