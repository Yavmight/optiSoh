"""Reference models: Gaussian Process Regression (GPR) and ElasticNet."""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import ElasticNet

def create_gpr_baseline():
    """Create GPR model with Matérn kernel for small-cell baseline."""
    kernel = Matern(nu=1.5) + WhiteKernel(noise_level=1.0)
    return GaussianProcessRegressor(kernel=kernel, random_state=42)

def create_elasticnet_baseline():
    return ElasticNet(random_state=42)
