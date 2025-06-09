# -------------------------------------------------------------------------
# file: hypersonic.py
# -------------------------------------------------------------------------
from dynamics import NonlinearSDE
from atmosphere import rho_nominal, DensityGRF, R_E
import casadi as ca
import numpy as np

# ---- physical constants (SI) --------------------------------------------
MU       = 3.986004418e14          # Earth's GM          
OMEGA_E  = 7.292115e-5             # sidereal rate       

# ---- Orbiter (Space Shuttle) geometry & aero ----------------------------
SLUG2KG  = 14.593902937            # exact US-to-SI
FT2M     = 0.3048
FT2_TO_M2 = FT2M**2

MASS = 7008 * SLUG2KG              # 102 274 kg
SREF = 2690 * FT2_TO_M2            # 249.91 m²

# C_L(α)  and  C_D(α)
CL_0   = -0.2070
CL_a   = 1.6760                    # per rad
CD_0   = 0.07854
CD_a   = -0.3529
CD_a2  = 2.0400

class HypersonicVehicle(NonlinearSDE):
    """
    ▸ 6-state planar shuttle model with density GRF only.
    ▸ NO additive process-noise:  g(x,u) ≡ 0  (stochasticity is multiplicative via ρ).
    """
    def __init__(self, rho_field: DensityGRF | None = None):
        super().__init__(dim_x=6, dim_u=2, dim_w=0)
        self.rho_field = rho_field     # if None → deterministic ρ̄(h)

    # ------------------------------------------------------------------ AERODYNAMICS
    @staticmethod
    def aero_coeffs(alpha: float) -> tuple[float, float]:
        cl = CL_0 + CL_a   * alpha
        cd = CD_0 + CD_a   * alpha + CD_a2 * alpha**2
        return cl, cd

    def aero_forces(self, h: float, v: float, alpha: float) -> tuple[float, float]:
        ρ = rho_nominal(h) if self.rho_field is None else self.rho_field(h)
        q = 0.5 * ρ * v**2
        cL, cD = self.aero_coeffs(alpha)
        L, D   = q * SREF * cL, q * SREF * cD
        return L, D

    # ------------------------------------------------------------------ EQUATIONS OF MOTION
    def f(self, x, u, t):
        r, _, φ, v, ψ, γ   = [x[i] for i in range(6)]        # states
        σ, α               = [u[i] for i in range(2)]        # controls: bank, angle-of-attack

        h        = r - R_E
        L, D     = self.aero_forces(h, v, α)
        a_s      = -D / MASS
        a_n      =  L * ca.cos(σ) / MASS
        a_w      =  L * ca.sin(σ) / MASS
        g_r      = MU / r**2

        rdot  = v * ca.sin(γ)
        θdot  = v * ca.cos(γ) * ca.sin(ψ) / (r * ca.cos(φ))
        φdot  = v * ca.cos(γ) * ca.cos(ψ) / r
        vdot  = (a_s - g_r * ca.sin(γ)
                 + (OMEGA_E**2) * r * ca.cos(φ) * (ca.sin(γ) * ca.cos(φ) - ca.cos(γ) * ca.sin(φ) * ca.cos(ψ)))
        ψdot  = (a_w / (v * ca.cos(γ)) - (v / r) * ca.cos(γ) * ca.tan(φ) * ca.cos(ψ)
                 - (OMEGA_E**2 * r / (v * ca.cos(γ))) * (ca.sin(φ) * ca.cos(φ) * ca.cos(ψ))
                      + 2 * OMEGA_E * (ca.tan(γ) * ca.cos(φ) * ca.sin(ψ) - ca.sin(φ)))
        
        γdot  = (a_n / v
                 - g_r * ca.cos(γ) / v
                 + v * ca.cos(γ) / r
                 + (OMEGA_E**2 * r / v) * ca.cos(φ)
                   * (ca.cos(γ) * ca.cos(φ) + ca.sin(γ) * ca.sin(φ) * ca.sin(ψ))
                 + 2 * OMEGA_E * ca.cos(ψ) * ca.cos(φ))

        return ca.vertcat(rdot, θdot, φdot, vdot, ψdot, γdot)
    
    def f_np(self, x, u, t):
        r, _, φ, v, ψ, γ   = x        # states
        σ, α               = u        # controls

        h        = max(r - R_E, 0.0)
        L, D     = self.aero_forces(h, v, α)
        a_s      = -D / MASS
        a_n      =  L * np.cos(σ) / MASS
        a_w      =  L * np.sin(σ) / MASS
        g_r      = MU / r**2

        rdot  = v * np.sin(γ)
        θdot  = v * np.cos(γ) * np.sin(ψ) / (r * np.cos(φ))
        φdot  = v * np.cos(γ) * np.cos(ψ) / r
        vdot  = (a_s - g_r * np.sin(γ)
                 + (OMEGA_E**2) * r * np.cos(φ)
                   * (np.sin(γ) * np.cos(φ) - np.cos(γ) * np.sin(φ) * np.sin(ψ)))
        ψdot  = (a_w / (v * np.cos(γ)) - (v / r) * np.cos(γ) * np.tan(φ) * np.cos(ψ)
                 - (OMEGA_E**2 * r / (v * np.cos(γ))) * (np.sin(φ) * np.cos(φ) * np.cos(ψ))
                      + 2 * OMEGA_E * (np.tan(γ) * np.cos(φ) * np.sin(ψ) - np.sin(φ)))
        
        γdot  = (a_n / v
                 - g_r * np.cos(γ) / v
                 + v * np.cos(γ) / r
                 + (OMEGA_E**2 * r / v) * np.cos(φ)
                   * (np.cos(γ) * np.cos(φ) + np.sin(γ) * np.sin(φ) * np.sin(ψ))
                 + 2 * OMEGA_E * np.cos(ψ) * np.cos(φ))

        return np.array([rdot, θdot, φdot, vdot, ψdot, γdot])

    # ------------ NO additive process noise ------------------------------------------------
    def g(self, x, u):
        return ca.MX.zeros((6, 0))       # zero-column matrix → deterministic SDE
