import numpy as np
from scipy.optimize import least_squares

# ============================================================
# All Required Functions (Self-Contained)
# ============================================================

def latent_field(Xbar, lam):
    Xbar = np.asarray(Xbar, dtype=float)
    if np.any(Xbar <= 0):
        raise ValueError("Allocation must have full support.")
    return -lam * np.log(Xbar)


def cross_task_separability_statistic(g, A, tau=1e-8):
    g = np.asarray(g, float)
    A = np.asarray(A, float)
    mask = A > tau
    ratio_resid = np.where(mask, 
                           (g - np.median(g / np.where(mask, A, 1), axis=0)) / 
                           np.where(mask, A, 1), 0)
    Rc = ratio_resid - ratio_resid.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Rc, compute_uv=False)
    return (s[0]**2) / (np.sum(s**2) + 1e-12)


def recover_params(Xbar, A, kappa):
    Xbar = np.asarray(Xbar, float)
    A = np.asarray(A, float)
    z = np.sum(A * Xbar, axis=0)
    m = len(z)

    def unpack(theta):
        alpha = 1.0 / (1.0 + np.exp(-theta[0]))
        gamma = np.empty(m)
        gamma[0] = 1.0
        gamma[1:] = np.exp(theta[1:])
        gamma = gamma / gamma.sum()
        return alpha, gamma

    def resid(theta):
        alpha, gamma = unpack(theta)
        denom = np.sum(gamma * z**alpha)
        pred = -gamma * z**(alpha - 1.0) / denom
        return pred - kappa

    theta0 = np.zeros(m)
    sol = least_squares(resid, theta0, method="trf", max_nfev=8000)
    alpha_hat, gamma_hat = unpack(sol.x)
    return float(alpha_hat), gamma_hat


def make_positive(n, m, lam, sigma, rng):
    A = rng.gamma(2.0, 1.0, size=(n, m))
    gamma = rng.gamma(2.0, 1.0, size=m)
    gamma /= gamma.sum()
    alpha = rng.uniform(0.3, 0.8)

    X = np.full((n, m), 1.0 / m)
    for _ in range(500):
        z = np.sum(A * X, axis=0)
        denom = np.sum(gamma * z**alpha)
        g = -(gamma * z**(alpha - 1.0))[None, :] * A / denom
        logits = -g / lam
        X = np.exp(logits - logits.max(axis=1, keepdims=True))
        X /= X.sum(axis=1, keepdims=True)

    X *= np.exp(sigma * rng.standard_normal((n, m)))
    X = np.clip(X, 1e-6, None)
    X /= X.sum(axis=1, keepdims=True)
    return X, A, alpha, gamma


def make_negative(n, m, lam, sigma, rng):
    A = rng.gamma(2.0, 1.0, size=(n, m))
    kappa = rng.uniform(-2, -0.5, size=m)
    extra = rng.standard_normal((n, m)) * 1.8
    g = A * kappa[None, :] + A * extra

    logits = -g / lam
    X = np.exp(logits - logits.max(axis=1, keepdims=True))
    X /= X.sum(axis=1, keepdims=True)

    X *= np.exp(sigma * rng.standard_normal((n, m)))
    X = np.clip(X, 1e-6, None)
    X /= X.sum(axis=1, keepdims=True)
    return X, A


def represent_test_heldout(Xbar, A, lam=1.0, B=1200, eta=0.10,
                           calib_frac=0.7, tau=1e-8, seed=42):
    rng = np.random.default_rng(seed)
    n = Xbar.shape[0]
    g = latent_field(Xbar, lam)

    idx = rng.permutation(n)
    n_calib = int(n * calib_frac)
    calib_idx = idx[:n_calib]
    test_idx = idx[n_calib:]

    g_calib = g[calib_idx]
    A_calib = A[calib_idx]
    g_test = g[test_idx]
    A_test = A[test_idx]

    kappa = np.array([
        np.median((g_calib[:, j] / A_calib[:, j])[A_calib[:, j] > tau])
        if np.any(A_calib[:, j] > tau) else 0.0
        for j in range(g.shape[1])
    ])

    S_obs = cross_task_separability_statistic(g_test, A_test, tau=tau)

    count = 0
    for _ in range(B):
        resid = g_test - A_test * kappa[None, :]
        RRp = np.column_stack([
            rng.permutation(resid[:, j]) for j in range(g.shape[1])
        ])
        g_perm = RRp + A_test * kappa[None, :]
        if cross_task_separability_statistic(g_perm, A_test, tau=tau) >= S_obs:
            count += 1
    p = count / B

    decision = "realizable" if p >= eta else "rejected"
    alpha_hat = gamma_hat = None
    if decision == "realizable":
        alpha_hat, gamma_hat = recover_params(Xbar, A, kappa)

    return {"p_value": p, "decision": decision}


# ============================================================
# Main Evaluation Across Multiple Noise Levels
# ============================================================

def evaluate_across_noise(n=80, m=4, lam=1.0, 
                          noise_levels=None, 
                          n_trials=200, B=1000, 
                          thresholds=None, seed=42):
    
    if noise_levels is None:
        noise_levels = [0.10, 0.15, 0.20, 0.25, 0.30]
    
    if thresholds is None:
        thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

    rng = np.random.default_rng(seed)
    
    print("=== Performance of represent_test_heldout across noise levels ===\n")

    for sigma in noise_levels:
        print(f"\n--- Noise Level: σ = {sigma} ---")

        # Collect p-values
        p_pos = []
        p_neg = []

        for _ in range(n_trials):
            Xp, Ap, _, _ = make_positive(n, m, lam, sigma, rng)
            res = represent_test_heldout(Xp, Ap, lam=lam, B=B, seed=int(rng.integers(1e9)))
            p_pos.append(res["p_value"])

            Xn, An = make_negative(n, m, lam, sigma, rng)
            res = represent_test_heldout(Xn, An, lam=lam, B=B, seed=int(rng.integers(1e9)))
            p_neg.append(res["p_value"])

        print(f"{'Threshold (η)':<12} {'Sensitivity':<12} {'Specificity':<12} {'Youden Index':<12}")
        print("-" * 50)

        best_eta = None
        best_youden = -1

        for eta in thresholds:
            sens = np.mean([1 if p >= eta else 0 for p in p_pos])
            spec = np.mean([1 if p < eta else 0 for p in p_neg])
            youden = sens + spec - 1

            print(f"{eta:<12.2f} {sens*100:>8.1f}%     {spec*100:>8.1f}%     {youden:>8.3f}")

            if youden > best_youden:
                best_youden = youden
                best_eta = eta

        print("-" * 50)
        print(f"Best threshold: η = {best_eta:.2f} (Youden = {best_youden:.3f})\n")


# ============================================================
# Run Evaluation
# ============================================================

if __name__ == "__main__":
    evaluate_across_noise(
        n=80, 
        m=4, 
        noise_levels=[0.10, 0.15, 0.20, 0.25], 
        n_trials=180, 
        B=1000
    )
