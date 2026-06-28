import numpy as np

# ============================================================
# Paste the helper functions + represent_test_parametric here
# (from the previous complete script)
# ============================================================

# [Copy the full code from the last response here: 
#  latent_field, separability_statistic_cv, make_positive, make_negative, 
#  and represent_test_parametric]

# ============================================================
# Function to evaluate performance across thresholds
# ============================================================

def find_operating_point(n=80, m=4, lam=1.0, sigma=0.20, n_trials=300, B=1200, 
                         thresholds=None, seed=42):
    if thresholds is None:
        thresholds = np.arange(0.02, 0.41, 0.02)   # from 0.02 to 0.40

    rng = np.random.default_rng(seed)
    
    results = []
    
    # Collect p-values for positives and negatives
    p_pos = []
    p_neg = []
    
    for _ in range(n_trials):
        # Positive control
        Xp, Ap, _, _ = make_positive(n, m, lam, sigma, rng)
        res = represent_test_parametric(Xp, Ap, lam=lam, B=B, seed=int(rng.integers(1e9)))
        p_pos.append(res["p_value"])
        
        # Negative control
        Xn, An = make_negative(n, m, lam, sigma, rng)
        res = represent_test_parametric(Xn, An, lam=lam, B=B, seed=int(rng.integers(1e9)))
        p_neg.append(res["p_value"])
    
    print(f"\n=== Operating Point Analysis (σ = {sigma}) ===\n")
    print(f"{'Threshold (η)':<12} {'Sensitivity':<12} {'Specificity':<12} {'Youden Index':<12}")
    print("-" * 50)
    
    best_threshold = 0
    best_score = -1
    
    for eta in thresholds:
        sens = np.mean([1 if p >= eta else 0 for p in p_pos])
        spec = np.mean([1 if p < eta else 0 for p in p_neg])
        youden = sens + spec - 1
        
        print(f"{eta:<12.2f} {sens*100:>8.1f}%     {spec*100:>8.1f}%     {youden:>8.3f}")
        
        if youden > best_score:
            best_score = youden
            best_threshold = eta
    
    print("-" * 50)
    print(f"\nBest operating point (by Youden's Index): η = {best_threshold:.2f}")
    print(f"Youden Index = {best_score:.3f}")
    
    # Also show performance at a few practical thresholds
    print("\nRecommended practical thresholds:")
    for eta in [0.05, 0.10, 0.15, 0.20]:
        sens = np.mean([1 if p >= eta else 0 for p in p_pos])
        spec = np.mean([1 if p < eta else 0 for p in p_neg])
        print(f"  η = {eta:.2f} → Sensitivity: {sens*100:.1f}%, Specificity: {spec*100:.1f}%")

# ============================================================
# Run the analysis
# ============================================================

if __name__ == "__main__":
    find_operating_point(sigma=0.20, n_trials=250, B=1000)
