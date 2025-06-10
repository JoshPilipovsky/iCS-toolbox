# ------------------------------------------------------------------------
# nominal_ocp.py          (drop into the same package as hypersonic.py)
# ------------------------------------------------------------------------
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from hypersonic import HypersonicVehicle, R_E
from atmosphere import rho_nominal
from metrics import dyn_pressure, heating_poly, normal_load

# === 1. Reference quantities for non-dimensionalisation ================
g0 = 9.80665  # surface gravity  [m/s²]
v0 = np.sqrt(R_E * g0)  #   ≈ 7.911 km/s
t0 = np.sqrt(R_E / g0)  #   ≈  803.6  s


def nd_state(x):
    """ dimensional → non-dimensional """
    r = x[0]
    th = x[1]
    ph = x[2]
    v = x[3]
    ps = x[4]
    ga = x[5]
    return ca.vertcat(r / R_E, th, ph, v / v0, ps, ga)


def d_state(x_nd):
    """ non-dimensional → dimensional """
    rn = x_nd[0]
    th = x_nd[1]
    ph = x_nd[2]
    vn = x_nd[3]
    ps = x_nd[4]
    ga = x_nd[5]
    return ca.vertcat(rn * R_E, th, ph, vn * v0, ps, ga)


# === 2. Discretisation parameters =======================================
N_ocp = 50  # number of shooting intervals
K_RUNG = 4  # RK4 sub-steps per interval


# === 3. Build OCP ========================================================
class NominalOCP:

    def __init__(self, x0, xf_mean, q_max, L_max, n_max, rho_field=None):

        self.model = HypersonicVehicle(rho_field=rho_field)

        # ----- optimisation variables ----------------------------------
        self.opti = ca.Opti()
        self.X = self.opti.variable(6, N_ocp + 1)  # non-dim states
        self.U = self.opti.variable(2, N_ocp + 1)  # controls  (σ,α)
        self.Tf = self.opti.variable()  # final time  (dim-less)

        dt_nd = self.Tf / N_ocp  # Δτ in dimensionless time

        # ----- warm-start: zero-order-hold control + straight-line state interp ----
        # # 1) control guess: hold bank & AoA = 0 on every interval
        # self.opti.set_initial(self.U, np.zeros((2, N)))

        # 1) warm-start guess for the controls:
        #    bank σ: linearly from –60° to 0°  (over N intervals)
        #    AoA  α: constant at +20°
        sigma0 = -60.0 * np.pi / 180  # start at –60°
        sigma1 = 0.0 * np.pi / 180  # end at   0°
        alpha0 = 20.0 * np.pi / 180  # constant AoA

        sigma_init = np.linspace(sigma0, sigma1, N_ocp + 1)
        alpha_init = np.full(N_ocp + 1, alpha0)

        # stack into the 2×N control array
        self.opti.set_initial(self.U, np.vstack([sigma_init, alpha_init]))

        # 2) straight-line interpolation between known x0 and xf for X[:,k]
        #    (fill “free” entries in xf_mean with x0 values for the guess)
        xf_full = x0.copy()
        xf_full[0] = xf_mean[0]  # rf -> given
        xf_full[3] = xf_mean[3]  # vf -> given
        xf_full[5] = xf_mean[5]  # γf -> given

        # now override the “free” entries with your specified endpoints:
        xf_full[1] = 85.0 * np.pi / 180  # θ final = +85 deg
        xf_full[2] = -30.0 * np.pi / 180  # φ final = -30 deg
        xf_full[4] = -80.0 * np.pi / 180  # ψ final = -80 deg

        x0_nd = nd_state(x0)
        xf_nd = nd_state(xf_full)

        for i in range(N_ocp + 1):
            k = i / N_ocp
            guess = (1 - k) * x0_nd + k * xf_nd
            self.opti.set_initial(self.X[:, i], guess)

        # ----- boundary conditions ------------------------------------
        self.opti.set_initial(self.X[:, 0], nd_state(x0))
        self.opti.subject_to(self.X[:, 0] == nd_state(x0))

        xT_target = xf_mean.copy()
        # θ, φ, ψ free  →  do not constrain  (set to None)
        for i, val in enumerate(xT_target):
            if val is not None:
                self.opti.subject_to(self.X[i, -1] == nd_state(
                    [val if j == i else 0 for j in range(6)])[i])

        # ----- dynamics via explicit RK4 ------------------------------
        def f_nd(x_nd, u):
            # unwrap, dimensionalise, propagate, then re-scale
            x_dim = d_state(x_nd)
            dx = self.model.f(x_dim, u, 0.0)  # nominal ρ̄
            return nd_state(dx) * (t0 / 1.0)  # because ẋ_nd = (t0)⁻¹ ẋ

        for k in range(N_ocp):
            xk = self.X[:, k]
            uk = self.U[:, k]
            h = dt_nd / K_RUNG
            x_next = xk
            for _ in range(K_RUNG):
                k1 = f_nd(x_next, uk)
                k2 = f_nd(x_next + 0.5 * h * k1, uk)
                k3 = f_nd(x_next + 0.5 * h * k2, uk)
                k4 = f_nd(x_next + h * k3, uk)
                x_next = x_next + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # ----- path constraints  (evaluate in dimensional variables) ---
        for k in range(N_ocp + 1):
            x_dim = d_state(self.X[:, k])

            r, _, _, v, _, _ = ca.vertsplit(x_dim)
            h = r - R_E
            ρ = rho_nominal(h)
            q = dyn_pressure(ρ, v)

            # Under a ZOH assumption, control u_k is applied on
            # [t_k, t_{k+1}). Pair each state with the control of the
            # same index, except for the terminal state which uses the
            # last control input.
            # if k < N_ocp:
            α_k = self.U[1, k]
            # else:
            #     α_k = self.U[1, -1]

            n = normal_load(ρ, v, α_k)
            Λ = heating_poly(ρ, v, α_k)

            self.opti.subject_to(q <= q_max)
            self.opti.subject_to(Λ <= L_max)
            self.opti.subject_to(n <= n_max)

        # ----- cost:    J = ∫ ||u||² dt  + w_T * Tf  -------------------
        wT = 1e-3
        integrand = ca.sum2(ca.sum1(self.U**2))
        self.opti.minimize(integrand * dt_nd + wT * self.Tf)

        # ----- variable bounds & initial guesses -----------------------
        self.opti.set_initial(self.Tf, 2200 / t0)  # ≈ 2200 s
        self.opti.subject_to(self.Tf >= 1000 / t0)
        self.opti.subject_to(self.Tf <= 4000 / t0)

        self.opti.subject_to(
            self.opti.bounded(-np.pi / 2, self.U[0, :],
                              np.pi / 2))  # α ∈ [−90°,+90°]
        self.opti.subject_to(
            self.opti.bounded(-np.pi / 2, self.U[1, :], np.pi / 2))

        # IPOPT options
        p_opts = dict(print_time=True)
        s_opts = dict(max_iter=5000, print_level=5)
        self.opti.solver("ipopt", p_opts, s_opts)

        # Flags for saving solution
        self._solved = False
        self._cache = {}

    # ==================================================================
    def solve(self, use_cache=False):
        """
        If use_cache=True and we’ve already solved once, just return the cached data.
        Otherwise run the optimization, cache, and return.
        """
        if use_cache and self._solved:
            return (self._cache['t'], self._cache['X'], self._cache['U'],
                    self._cache['Tf'])

        # Solve OCP
        sol = self.opti.solve()

        # Process optimal solution
        t_nd = np.linspace(0, sol.value(self.Tf), N_ocp + 1)
        t_dim = t_nd * t0
        Tf_out = sol.value(self.Tf) * t0

        # pull out the full non-dimensional state: shape = (6, N+1)
        X_nd = sol.value(self.X)

        # un-scale to dimensional:
        #    row 0 = r/R_e -> r = R_E * (row0)
        #    row 3 = v/v0  -> v = v0  * (row3)
        X_out = np.zeros_like(X_nd)
        X_out[0, :] = X_nd[0, :] * R_E
        X_out[1, :] = X_nd[1, :]  # θ
        X_out[2, :] = X_nd[2, :]  # φ
        X_out[3, :] = X_nd[3, :] * v0
        X_out[4, :] = X_nd[4, :]  # ψ
        X_out[5, :] = X_nd[5, :]  # γ

        # get the optimal control: shape = (2, N)
        U_out = sol.value(self.U)

        # cache solution:
        self._cache = dict(t=t_dim, X=X_out, U=U_out, Tf=Tf_out)
        self._solved = True

        return t_dim, X_out, U_out, Tf_out

    def save_solution(self, filepath: str):
        """ Save the last solution to a .npz file. """
        if not self._solved:
            raise RuntimeError("No solution to save... run solve() first.")
        np.savez(filepath,
                 t=self._cache['t'],
                 X=self._cache['X'],
                 U=self._cache['U'],
                 Tf=self._cache['Tf'])

    @classmethod
    def load_solution(cls, filepath: str):
        """
        Load a saved solution from disk into a dummy OCP object.
        You can then do ocp._cache[...] or directly plot.
        """
        data = np.load(filepath)
        obj = cls.__new__(cls)  # bypass __init__
        obj._solved = True
        obj._cache = dict(
            t=data['t'],
            X=data['X'],
            U=data['U'],
            Tf=data['Tf'].item()  # scalar
        )
        return obj


def simulate_nominal(x0, sigma_init, alpha_init, Tf, N, rho_field=None):
    """
    Simulate ẋ = f(x,u,t) with piecewise-constant controls (ZOH).
    - x0:        initial 6×1 state (dimensional)
    - sigma_init, alpha_init: arrays of length N
    - Tf:        total time [s]
    - N:         number of intervals
    Returns t_grid (N+1), X_sim (N+1×6).
    """
    model = HypersonicVehicle(rho_field=rho_field)

    # time grid
    t_grid = np.linspace(0, Tf, N + 1)

    # control lookup: returns [σ,α] at time t
    def u_of_t(t):
        k = min(int(t / Tf * N), N - 1)
        return np.array([sigma_init[k], alpha_init[k]])

    # ODE right-hand side for solve_ivp
    def ode(t, x):
        u = u_of_t(t)
        dx = model.f_np(x, u, t)  # uses nominal ρ
        return dx

    sol = solve_ivp(ode, (0, Tf), x0, t_eval=t_grid, rtol=1e-8, atol=1e-8)
    return sol.t, sol.y


# ============== quick demo / sanity check ==============================
if __name__ == "__main__":
    from scenario import x0_mean, xf_mean, q_max, Λ_max, n_max
    deg2rad = np.pi / 180
    m2ft = 3.28084
    Wm2_to_Btu = 1 / 1055.06
    factor = Wm2_to_Btu / m2ft**2
    N = 5000
    Tf_guess = 2200.0

    sigma_init = np.linspace(-60 * deg2rad, 0, N)
    alpha_init = np.full(N, 20 * deg2rad)

    t_sim, X_sim = simulate_nominal(x0_mean, sigma_init, alpha_init, Tf_guess,
                                    N)

    # 1) State trajectories (altitude & speed example)
    alt_m = X_sim[0, :] - R_E
    alt_kft = alt_m * m2ft / 1000.0
    vel_mps = X_sim[3, :]
    vel_kftps = vel_mps * m2ft / 1000.0
    # theta_deg = X_sim[1, :] / deg2rad
    # phi_deg = X_sim[2, :] / deg2rad
    # psi_deg = X_sim[4, :] / deg2rad
    # gamma_deg = X_sim[5, :] / deg2rad
    # t_min = t_sim / 60

    # fig, axs = plt.subplots(3, 2, figsize=(8, 9), sharex=True)
    # axs = axs.flatten()
    # series = [alt_kft, theta_deg, phi_deg, vel_kftps, psi_deg, gamma_deg]
    # labels = [
    #     "Altitude [kft]", "θ [deg]", "φ [deg]", "Speed [kft/s]", "ψ [deg]",
    #     "γ [deg]"
    # ]

    # for i, ax in enumerate(axs):
    #     ax.plot(t_min, series[i])
    #     ax.set_ylabel(labels[i])
    #     ax.grid(True)
    #     if i >= 4:  # only bottom row
    #         ax.set_xlabel("Time [min]")

    # # 2) Path constraints
    # rho = rho_nominal(X_sim[0, :] - R_E)
    # q = dyn_pressure(rho, X_sim[3, :])
    # Lam = heating_poly(rho, X_sim[3, :],
    #                    np.hstack([alpha_init, alpha_init[-1]]))
    # n = normal_load(rho, X_sim[3, :], np.hstack([alpha_init, alpha_init[-1]]))

    # fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    # axs[0].plot(t_min, q)
    # axs[0].axhline(q_max, color="r", linestyle="--")
    # axs[0].set_ylabel("q [Pa]")

    # axs[1].plot(t_min, Lam * factor)
    # axs[1].axhline(Λ_max * factor, color="r", linestyle="--")
    # axs[1].set_ylabel("Λ [W/m²]")

    # axs[2].plot(t_min, n)
    # axs[2].axhline(n_max, color="r", linestyle="--")
    # axs[2].set_ylabel("n [g]")
    # axs[2].set_xlabel("Time [min]")

    # for ax in axs:
    #     ax.grid(True)
    # plt.tight_layout()
    # plt.show()

    # -------------------------- #

    # Define OCP (obj)
    ocp = NominalOCP(x0_mean, xf_mean, q_max, Λ_max, n_max)

    # Solve OCP
    t, X, U, Tf = ocp.solve()

    # Save solution
    ocp.save_solution("shuttle_nominal.npz")

    # # --- OPTIONAL --- [load solution]
    # ocp = NominalOCP.load_solution("shuttle_nominal.npz")
    # t, X, U, Tf = ocp._cache['t'], ocp._cache['X'], ocp._cache['U'], ocp._cache['Tf']

    # ------------ basic plots -----------------------------------------
    # 1) Plot all 6 state trajectories with units adjusted
    alt_m = X[0, :] - R_E
    alt_kft = alt_m * m2ft / 1000.0
    vel_mps = X[3, :]
    vel_kftps = vel_mps * m2ft / 1000.0
    theta_deg = X[1, :] / deg2rad
    phi_deg = X[2, :] / deg2rad
    psi_deg = X[4, :] / deg2rad
    gamma_deg = X[5, :] / deg2rad
    t_min = t / 60.0

    fig, axs = plt.subplots(3, 2, figsize=(6, 8), sharex=True)
    axs = axs.flatten()
    series = [alt_kft, theta_deg, phi_deg, vel_kftps, psi_deg, gamma_deg]
    labels = [
        "Altitude [kft]", "θ [deg]", "φ [deg]", "Speed [kft/s]", "ψ [deg]",
        "γ [deg]"
    ]

    for i, ax in enumerate(axs):
        ax.plot(t_min, series[i])
        ax.set_ylabel(labels[i])
        ax.grid(True)
        if i >= 4:  # only bottom row
            ax.set_xlabel("Time [min]")

    plt.tight_layout()
    plt.savefig("state_trajectories.png", dpi=150, bbox_inches='tight')
    print("✓ Saved state trajectories to state_trajectories.png")
    plt.close()

    # 2) Plot controls (ZOH, piecewise constant on each interval)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.step(t[:-1] / 60, U[0, :] * 180 / np.pi, where="post", label="Bank σ")
    ax2.step(t[:-1] / 60, U[1, :] * 180 / np.pi, where="post", label="AoA α")
    ax2.set_ylabel("Control [deg]")
    ax2.set_xlabel("Time [min]")
    ax2.set_title("Control Profiles (ZOH)")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.savefig("control_profiles.png", dpi=150, bbox_inches='tight')
    print("✓ Saved control profiles to control_profiles.png")
    plt.close()

    # 3) Plot constraints with limits
    rho = rho_nominal(X[0, :] - R_E)
    q = dyn_pressure(rho, X[3, :])

    # Under ZOH, control U[:,k] is active on [t_k, t_{k+1}). Attach the
    # control of the same index to each state, except that the final
    # state uses the last control input.
    Λ = np.zeros(len(t))
    n = np.zeros(len(t))
    for k in range(len(t)):
        # if k < len(t) - 1:
        alpha = U[1, k]
        # else:
        #     alpha = U[1, -1]
        Λ[k] = heating_poly(rho[k], X[3, k], alpha)
        n[k] = normal_load(rho[k], X[3, k], alpha)

    fig3, ax3 = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    ax3[0].plot(t / 60, q, label='q(t)')
    ax3[0].axhline(q_max, color='r', linestyle='--', label='q_max')
    ax3[0].set_ylabel("q [Pa]")
    ax3[0].grid(True)
    ax3[0].legend()

    ax3[1].plot(t / 60, Λ * factor, label='Λ(t)')
    ax3[1].axhline(Λ_max * factor, color='r', linestyle='--', label='Λ_max')
    ax3[1].set_ylabel("Heating Rate [Btu/ft²·s]")
    ax3[1].grid(True)
    ax3[1].legend()

    ax3[2].plot(t / 60, n, label='n(t)')
    ax3[2].axhline(n_max, color='r', linestyle='--', label='n_max')
    ax3[2].set_ylabel("Load Factor [g]")
    ax3[2].set_xlabel("Time [min]")
    ax3[2].grid(True)
    ax3[2].legend()

    fig3.suptitle("Path Constraints vs. Time")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("path_constraints.png", dpi=150, bbox_inches='tight')
    print("✓ Saved path constraints to path_constraints.png")
    plt.close()
    
    print(f"\n🎯 Optimization completed successfully!")
    print(f"   Final time: {Tf:.1f} seconds ({Tf/60:.1f} minutes)")
    print(f"   Solution saved to: shuttle_nominal.npz")
    print(f"   Plots saved as PNG files in the current directory")
