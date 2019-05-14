import itertools

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import linalg
from sklearn import mixture
from sklearn.model_selection import train_test_split
from hmmlearn.hmm import GaussianHMM
import tf.transformations as T
from gmr.utils import check_random_state
from gmr import MVN, GMM, plot_error_ellipses, plot_error_ellipses3d, plot_axes

if __name__ == '__main__':
    X = np.loadtxt("./sample1.txt")
    gmm = GMM(n_components=3, random_state=0)

    # gmm.from_samples(X[:, [0, 2]])
    # plt.subplot(1, 1, 1)
    # plt.scatter(X[:, 0], X[:, 2]) # X-Z plane
    # X_test = np.linspace(0, 2 * np.pi, 100)
    # Y = gmm.predict(np.array([1]), X_test[:, np.newaxis])
    # plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b"])

    gmm.from_samples(X[:, :])
    from mpl_toolkits.mplot3d import Axes3D
    plt.subplot(1, 1, 1, projection='3d')
    plt.gca().scatter3D(X[:, 0], X[:, 1], X[:, 2])
    # plt.gca().set_xlim(0, 400)
    # plt.gca().set_ylim(-200, 200)
    # plt.gca().set_zlim(0, 400)
    plot_error_ellipses3d(plt.gca(), gmm, colors=["r", "g", "b"])
    plot_axes(plt.gca(), gmm, colors=["r", "g", "b"], factor=10.0)

    # x, y = np.meshgrid(np.linspace(-10, 300, 100), np.linspace(-10, 200, 100))
    # X_test = np.vstack((x.ravel(), y.ravel())).T
    # p = gmm.to_probability_density(X_test)
    # p = p.reshape(*x.shape)
    # plt.contourf(x, y, p)
    # X_sampled = gmm.sample(100)
    plt.show()
