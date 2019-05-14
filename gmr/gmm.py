import numpy as np
from .utils import check_random_state
from .mvn import MVN


class GMM(object):
    """Gaussian Mixture Model.

    Parameters
    ----------
    n_components : int
        Number of MVNs that compose the GMM.

    priors : array, shape (n_components,), optional
        Weights of the components.

    means : array, shape (n_components, n_features), optional
        Means of the components.

    covariances : array, shape (n_components, n_features, n_features), optional
        Covariances of the components.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.
    """
    def __init__(self, n_components, priors=None, means=None, covariances=None,
                 verbose=0, random_state=None):
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    def _check_initialized(self):
        if self.priors is None:
            raise ValueError("Priors have not been initialized")
        if self.means is None:
            raise ValueError("Means have not been initialized")
        if self.covariances is None:
            raise ValueError("Covariances have not been initialized")

    def from_samples(self, X, R_diff=1e-4, n_iter=100):
        """MLE of the mean and covariance.

        Expectation-maximization is used to infer the model parameters. The
        objective function is non-convex. Hence, multiple runs can have
        different results.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples from the true function.

        R_diff : float
            Minimum allowed difference of responsibilities between successive
            EM iterations.

        n_iter : int
            Maximum number of iterations.

        Returns
        -------
        self : MVN
            This object.
        """
        n_samples, n_features = X.shape

        if self.priors is None:
            self.priors = np.ones(self.n_components,
                                  dtype=np.float) / self.n_components

        if self.means is None:
            # TODO k-means++
            indices = self.random_state.choice(
                np.arange(n_samples), self.n_components)
            self.means = X[indices]

        if self.covariances is None:
            self.covariances = np.empty((self.n_components, n_features,
                                         n_features))
            for k in range(self.n_components):
                self.covariances[k] = np.eye(n_features)

        R = np.zeros((n_samples, self.n_components))
        for _ in range(n_iter):
            R_prev = R

            # Expectation
            R = self.to_responsibilities(X)

            if np.linalg.norm(R - R_prev) < R_diff:
                if self.verbose:
                    print("EM converged.")
                break

            # Maximization
            w = R.sum(axis=0)
            R_n = R / w
            self.priors = w / w.sum()
            self.means = R_n.T.dot(X)
            for k in range(self.n_components):
                Xm = X - self.means[k]
                self.covariances[k] = (R_n[:, k, np.newaxis] * Xm).T.dot(Xm)

        return self

    def sample(self, n_samples):
        """Sample from Gaussian mixture distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the GMM.
        """
        self._check_initialized()

        mvn_indices = self.random_state.choice(
            self.n_components, size=(n_samples,), p=self.priors)
        mvn_indices.sort()
        split_indices = np.hstack(
            ((0,), np.nonzero(np.diff(mvn_indices))[0] + 1, (n_samples,)))
        clusters = np.unique(mvn_indices)
        lens = np.diff(split_indices)
        samples = np.empty((n_samples, self.means.shape[1]))
        for i, (k, n_samples) in enumerate(zip(clusters, lens)):
            samples[split_indices[i]:split_indices[i + 1]] = MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state).sample(n_samples=n_samples)
        return samples

    def to_responsibilities(self, X):
        """Compute responsibilities of each MVN for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        R : array, shape (n_samples, n_components)
        """
        self._check_initialized()

        n_samples = X.shape[0]
        R = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            R[:, k] = self.priors[k] * MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state).to_probability_density(X)
        R_norm = R.sum(axis=1)[:, np.newaxis]
        R_norm[np.where(R_norm == 0.0)] = 1.0
        R /= R_norm
        return R

    def to_probability_density(self, X):
        """Compute probability density.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        p : array, shape (n_samples,)
            Probability densities of data.
        """
        self._check_initialized()

        p = [MVN(mean=self.means[k], covariance=self.covariances[k],
                 random_state=self.random_state).to_probability_density(X)
             for k in range(self.n_components)]
        return np.dot(self.priors, p)

    def condition(self, indices, x):
        """Conditional distribution over given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : array, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        conditional : GMM
            Conditional GMM distribution p(Y | X=x).
        """
        self._check_initialized()

        n_features = self.means.shape[1] - len(indices)
        priors = np.empty(self.n_components)
        means = np.empty((self.n_components, n_features))
        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                      random_state=self.random_state)
            conditioned = mvn.condition(indices, x)
            priors[k] = (self.priors[k] *
                         mvn.marginalize(indices).to_probability_density(x))
            means[k] = conditioned.mean
            covariances[k] = conditioned.covariance
        priors /= priors.sum()
        return GMM(n_components=self.n_components, priors=priors, means=means,
                   covariances=covariances, random_state=self.random_state)

    def predict(self, indices, X):
        """Predict means of posteriors.

        Same as condition() but for multiple samples.

        Parameters
        ----------
        indices : array, shape (n_features_1,)
            Indices of dimensions that we want to condition.

        X : array, shape (n_samples, n_features_1)
            Values of the features that we know.

        Returns
        -------
        Y : array, shape (n_samples, n_features_2)
            Predicted means of missing values.
        """
        self._check_initialized()

        n_samples, n_features_1 = X.shape
        n_features_2 = self.means.shape[1] - n_features_1
        Y = np.empty((n_samples, n_features_2))
        for n in range(n_samples):
            conditioned = self.condition(indices, X[n])
            Y[n] = conditioned.priors.dot(conditioned.means)
        return Y

    def to_ellipses(self, factor=1.0):
        """Compute error ellipses.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        ellipses : array, shape (n_components, 3)
            Parameters that describe the error ellipses of all components:
            angles, widths and heights.
        """
        self._check_initialized()

        res = []
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                      random_state=self.random_state)
            res.append((self.means[k], mvn.to_ellipse(factor)))
        return res

    def to_ellipses3d(self, factor=1.0):
        """Compute error ellipses.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        ellipses : array, shape (n_components, 3)
            Parameters that describe the error ellipses of all components:
            angles, widths and heights.
        """
        self._check_initialized()

        res = []
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                      random_state=self.random_state)
            res.append((self.means[k], mvn.to_ellipse3d(factor)))
        return res

    def to_axes(self, factor=1.0):
        self._check_initialized()

        res = []
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                      random_state=self.random_state)
            res.append(mvn.to_axis(factor))
        return res


def plot_error_ellipses(ax, gmm, colors=None):
    """Plot error ellipses of GMM components.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    gmm : GMM
        Gaussian mixture model.
    """
    from matplotlib.patches import Ellipse
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    for factor in np.linspace(0.5, 4.0, 8):
        for mean, (angle, width, height) in gmm.to_ellipses(factor):
            ell = Ellipse(xy=mean, width=width, height=height,
                          angle=np.degrees(angle))
            ell.set_alpha(0.25)
            if colors is not None:
                ell.set_color(next(colors))
            ax.add_artist(ell)

def plot_error_ellipses3d(ax, gmm, colors=None):
    """Plot error ellipses of GMM components.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    gmm : GMM
        Gaussian mixture model.
    """
    from matplotlib.patches import Ellipse
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    for factor in [1.0]: # np.linspace(0.5, 4.0, 8):
        for mean, (angles, dims) in gmm.to_ellipses3d(factor):
            x, y, z = Ellipse3d(mean=mean, dims=dims,
                                angles=angles)
            if colors is not None:
                ax.plot_surface(x, y, z, color=next(colors), alpha=0.3)
            else:
                ax.plot_surface(x, y, z, alpha=0.25)

def plot_axes(ax, gmm, colors=None, factor=1.0):
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    for mean, axis in zip(gmm.means, gmm.to_axes(factor)):
    # for axis in gmm.to_axes(factor):
        for a in axis:
            # x, y, z = a[0], a[1], a[2]
            XYZ = np.concatenate([mean, mean+a]).reshape((2, -1)).T
            x = XYZ[0]
            y = XYZ[1]
            z = XYZ[2]
            # print(x, y, z)
            if colors is not None:
                ax.plot(x, y, z, "-", linewidth=5.0, color=next(colors), alpha=0.3)
            else:
                ax.plot(x, y, z, "-", linewidth=5.0, alpha=0.25)

def Ellipse3d(mean, dims, angles, num=100):
    """
    Plot an 3d ellipse
    """
    uu = np.linspace(0, 2 * np.pi, num)
    vv = np.linspace(0, np.pi, num)
    x = dims[0] * np.outer(np.cos(uu), np.sin(vv))
    y = dims[1] * np.outer(np.sin(uu), np.sin(vv))
    z = dims[2] * np.outer(np.ones_like(uu), np.cos(vv))
    xyz = np.vstack([x.reshape(1,-1), y.reshape(1,-1), z.reshape(1,-1)])
    theta, phi = angles
    R_theta = np.array([[np.cos(theta), np.sin(theta), 0],
                        [-np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    R_phi = np.array([[1, 0, 0],
                      [0, np.cos(phi), np.sin(phi)],
                      [0, -np.sin(phi), np.cos(phi)]])
    R = np.dot(R_phi, R_theta)
    xyz = np.dot(R, xyz)
    x = xyz[0].reshape(num, num)
    y = xyz[1].reshape(num, num)
    z = xyz[2].reshape(num, num)
    x = x + mean[0]
    y = y + mean[1]
    z = z + mean[2]
    return x, y, z

