"""
Very-lightweight wrapper around a 1-D exponential ‘mean’ profile
  ρ̄(h) = ρ0 · exp(-(h-h0)/H_s)
with a stationary Gaussian random field perturbation
  δρ(h) ~ GP(0, Σ(h1,h2)).

The kernel reproduces the piecewise–variance model you described.
"""
import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist

# --- canonical constants (SI) ---------------------------------------------
R_E     = 6_371_000.0                     # mean Earth radius   [m]
RHO0    = 1.225                           # sea-level density   [kg·m⁻³]
H_SCALE = 8_500.0                         # scale height        [m]

# GRF hyper-parameters (tune per mission)
H_TRANS       = 30_000.0                  # transition altitude [m]
SIGMA_RHO_MAX = 0.30                      # 30 % 1-σ envelope  (≈Earth-GRAM)
C_S           = 8_000.0                   # controls variance rise rate

def rho_nominal(h: np.ndarray) -> np.ndarray:
    """ISO “1976” style exponential atmosphere (valid to ≈90 km)."""
    return RHO0 * np.exp(-(h) / H_SCALE)

def rho_kernel(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Σ(h1,h2) from the piecewise formula the user supplied."""
    h1 = np.atleast_1d(h1); h2 = np.atleast_1d(h2)
    Δ   = np.abs(cdist(h1[:,None], h2[:,None], metric='euclidean'))
    h_min = np.minimum.outer(h1, h2)

    var = np.where(
        h_min < H_TRANS,
        SIGMA_RHO_MAX**2 *
        np.exp((h_min - H_TRANS) / C_S),
        SIGMA_RHO_MAX**2
    )
    return var * np.exp(-Δ / H_SCALE)

class DensityGRF:
    """Generate correlated δρ samples on an altitude grid."""
    def __init__(self, h_grid, rng=None):
        self.h = np.asarray(h_grid)
        self.Sigma = rho_kernel(self.h, self.h)
        self.L = np.linalg.cholesky(self.Sigma + 1e-12*np.eye(len(self.h)))
        self.rng = default_rng(rng)

    def sample(self) -> np.ndarray:
        return self.L @ self.rng.standard_normal(len(self.h))

    def full(self) -> np.ndarray:
        """One realisation of total density ρ(h) = ρ̄(h)(1+δρ)."""
        δρ = self.sample()
        return rho_nominal(self.h) * (1.0 + δρ)
