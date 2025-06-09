import numpy as np
from hypersonic import R_E

deg2rad = np.pi / 180
ft2m    = 0.3048

# initial state  (perfect knowledge ⇒ mean state = state value)
h0   = 260_000 * ft2m                       # 79 248 m
v0   = 24_061 * ft2m                       # 7 335 m/s
γ0   = -1.064 * deg2rad                    # −0.01857 rad

x0_mean = np.array([R_E + h0,
                    0.0,                   # θ
                    0.0,                   # φ
                    v0,
                    0.0,                   # ψ
                    γ0])

# terminal mean targets
hT   = 80_000 * ft2m                        # 24 384 m
vT   = 2_500 * ft2m                        # 762 m/s
γT   = -5.0 * deg2rad                      # −0.08727 rad

xf_mean = np.array([R_E + hT,
                    None,                  # free longitude
                    None,                  # free latitude
                    vT,
                    None,                  # free heading
                    γT])

# constraint scalars (literature values, Shuttle-specific)
q_max       = 819 * 47.8803               # 819 psf ⇒ 3.9 × 10⁴ Pa
Λ_max       = 70  * 1055.06 * 3.28084**2  # 70 Btu/ft²/s ⇒ 7.95 × 10⁵ W/m²
n_max       = 2.5                         # Orbiter structural limit
