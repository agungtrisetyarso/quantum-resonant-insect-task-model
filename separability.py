import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ============================================================
# Diagnostic: Raw ROC on the CV separability statistic
# (No bootstrap, no calibration — pure signal check)
# ============================================================

np.random.seed(42)
n_agents = 100
n_tasks = 4
n_trials = 500
noise_levels = [0.08, 0.18, 0.35]
labels = ['Low noise (σ=0.08)', 'Medium noise (σ=0.18)', 'High noise (σ=0.35)']

plt.figure(figsize=(8, 6))
colors = ['#2E86AB', '#E94F37', '#44AF69']

for i, sigma in enumerate(noise_levels):
    y_true = []
    scores = []          # We will use -CV so higher = more separable
    
    for _ in range(n_trials):
        # Positive (AR-realizable)
        Xp, Ap, _, _ = make_positive(n_agents, n_tasks, lam=1.0, sigma=sigma, 
                                     rng=np.random.default_rng())
        gp = latent_field(Xp, lam=1.0)
        cv_pos = separability_statistic_cv(gp, Ap)   # lower CV = more separable
        scores.append(-cv_pos)
        y_true.append(1)
        
        # Negative (non-realizable)
        Xn, An = make_negative(n_agents, n_tasks, lam=1.0, sigma=sigma,
                               rng=np.random.default_rng())
        gn = latent_field(Xn, lam=1.0)
        cv_neg = separability_statistic_cv(gn, An)
        scores.append(-cv_neg)
        y_true.append(0)
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=colors[i], lw=2.5, 
             label=f'{labels[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Raw ROC on CV Separability Statistic\n(No Calibration)', fontsize=13)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('raw_roc_diagnostic.png', dpi=300, bbox_inches='tight')
plt.show()
