# Code for "Exact damping laws for threshold-based task allocation with finite behavioural response time"

A. Trisetyarso and T. Taufikurahman, submitted to *Mathematical Biosciences*.

All scripts are plain Python 3 (tested with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11).
Random seeds are fixed, so every number in the manuscript is reproduced exactly.

```
pip install -r requirements.txt
python variance_law_check.py   # Thm 2 and Cor 1: variance law to ~1e-10, probit inequality
python fig_variance_law.py     # Fig. 8 (variance law, scaling triad, stimulus-noise invariance)  ~25 s
python reversal.py             # Table 3 (inhibition floor and heterogeneity reversal)            <5 s
python cooling.py              # Cor. 4 (stimulus noise with passive decay), ABM vs closed form   ~20 s
python finite_size.py          # Table 7 (ABM vs large-N vs O(1/N) system-size correction)        ~1 min
```

| Manuscript item | Script |
|---|---|
| Theorem 2, Corollary 1 (numerical check) | `variance_law_check.py` |
| Fig. 8, Theorem 4 checks | `fig_variance_law.py` |
| Table 3, Theorem 3 checks | `reversal.py` |
| Corollary 4 checks (Section 3.4) | `cooling.py` |
| Table 7, Eq. (emre), Section 4.2 | `finite_size.py` |
| Figs. 1-7, Table 2 | `original_scripts/` (model.py, figs_v2.py, fig_sources.py, check_N.py) |

Units: time in units of the response time tau, stimulus in units of the median threshold;
the feedback gain is A = alpha*tau/theta_bar.

Licence: MIT (see LICENSE). Cite via CITATION.cff.
