""" Source: https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html """
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, *, nsig, ax=None, **kwargs):
	"""Draw an ellipse with a given position and covariance"""
	ax = ax or plt.gca()

	# Convert covariance to principal axes
	if covariance.shape == (2, 2):
		U, s, Vt = np.linalg.svd(covariance)
		angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
		width, height = 2 * np.sqrt(s)
	else:
		angle = 0
		width, height = 2 * np.sqrt(covariance)

	# Draw the Ellipse
	for sig_factor in (np.arange(nsig) + 1):
		ax.add_patch(Ellipse(position,
			sig_factor * width, sig_factor * height,
			angle, **kwargs))

def plot_gmm(gmm, X=None, *, nsig=4, label=True, ax=None):
	ax = ax or plt.gca()

	if X is not None:
		labels = gmm.predict(X)
		if label:
			ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
		else:
			ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
		ax.axis('equal')

	w_factor = 0.2 / gmm.weights_.max()
	_params = zip(gmm.means_, gmm.covariances_, gmm.weights_)
	for i, (pos, covar, w) in enumerate(_params):
		ax.scatter(*pos, marker="x", color="black")
		ax.text(*pos, s=f"Comp #{i}",
			bbox=dict(facecolor="white", alpha=0.5),
			horizontalalignment="center",
			verticalalignment="center",
		)
		draw_ellipse(pos, covar, nsig=nsig, ax=ax, alpha=w * w_factor)
	ax.autoscale_view()
	return ax
