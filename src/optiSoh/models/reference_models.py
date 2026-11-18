"""Reference models: Gaussian Process Regression (GPR) and ElasticNet."""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import ElasticNet


def create_gpr_baseline(random_state: int = 42) -> GaussianProcessRegressor:
    """Create GPR model with Matérn kernel for small-cell baseline."""
    kernel = Matern(nu=1.5) + WhiteKernel(noise_level=1.0)
    return GaussianProcessRegressor(kernel=kernel, random_state=random_state)


def create_elasticnet_baseline(random_state: int = 42) -> ElasticNet:
    return ElasticNet(random_state=random_state)
