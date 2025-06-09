# -------------------------------------------------------------------------
# file: metrics.py            (new helper)
# -------------------------------------------------------------------------
import numpy as np
from hypersonic import SREF, MASS, CL_0, CL_a, CD_0, CD_a, CD_a2

# ---------- dynamic pressure ------------------------------------------------
def dyn_pressure(rho: float, v: float) -> float:
    return 0.5 * rho * v**2          # [Pa]

# ---------- aerodynamic forces ---------------------------------------------
def aero_coeffs(alpha: float) -> tuple[float, float]:
    cl = CL_0 + CL_a   * alpha
    cd = CD_0 + CD_a   * alpha + CD_a2 * alpha**2
    return cl, cd

def lift_drag(rho: float, v: float, alpha: float) -> tuple[float, float]:
    q       = dyn_pressure(rho, v)
    cL, cD  = aero_coeffs(alpha)
    return q * SREF * cL, q * SREF * cD

# ---------- normal load factor n = √(L²+D²) / (m g) ------------------------
G0 = 9.80665
def normal_load(rho: float, v: float, alpha: float) -> float:
    L, D = lift_drag(rho, v, alpha)
    return np.sqrt(L**2 + D**2) / (MASS * G0)

# ---------- heating-rate models -------------------------------------------
#  (a) Sutton–Graves-style: Λ = kλ √ρ v³
K_LAMBDA = 0.4369e-5               # kg^(1/2)·m^(−3/2)

def heating_simple(rho: float, v: float) -> float:
    return K_LAMBDA * np.sqrt(rho) * v**3      # [W/m²]

#  (b) Exponent-3.07 model with α-dependent prefactor (converted to SI)
import math
K0   = 9.289e-9                    # imperial constant
BTU2J   = 1055.06
FT2M    = 0.3048
SLUG2KG = 14.593902937

CONST_SI = K0 * BTU2J / (FT2M**3.57) / math.sqrt(SLUG2KG)  # 1.78 ×10⁻⁴

C0, C1, C2, C3 = 1.067, -1.101, 0.6988, -0.1903

def C_alpha(alpha: float) -> float:
    return CONST_SI * (C0 + C1*alpha + C2*alpha**2 + C3*alpha**3)

def heating_poly(rho: float, v: float, alpha: float) -> float:
    return C_alpha(alpha) * np.sqrt(rho) * v**3.07           # [W/m²]
