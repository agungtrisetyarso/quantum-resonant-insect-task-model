cd ~/Projects/insect-task-model

cat > generate_all_figures.py << 'EOF'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import qutip as qt

# ====================== DATA LOADING & CLEANING ======================
df = pd.read_csv("~/Downloads/QuantumBiology/Projects/IWants.csv")
activity = df["RMSD"].values
valid = np.isfinite(activity)          # remove NaNs / infs
activity = activity[valid]
time = np.arange(len(activity))

print(f"✅ Data loaded and cleaned — {len(activity)} frames")

# ====================== FIG. 1 — Empirical colony activity ======================
plt.figure(figsize=(12, 5))
plt.plot(time, activity, label="Empirical Colony Activity (RMSD)", linewidth=1.5)
peaks, _ = find_peaks(activity, distance=8, prominence=0.1)
plt.plot(time[peaks], activity[peaks], "x", markersize=8, label="Detected Peaks")
plt.xlabel("Time (frames)")
plt.ylabel("Activity level")
plt.title("Fig. 1 — Ooceraea biroi colony activity — IW experiment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Fig1_Empirical_Activity.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Fig. 1 saved as Fig1_Empirical_Activity.png")

# ====================== FIG. 2 — Resonant LC/RLC fit ======================
def damped_osc(t, A, omega, gamma, phi, C):
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi) + C

popt, _ = curve_fit(damped_osc, time, activity,
                    p0=[3, 0.1, 0.01, 0, np.mean(activity)])

print("Fitted RLC parameters (A, ω, γ, φ, C):", popt)

plt.figure(figsize=(12, 5))
plt.plot(time, activity, label="Empirical Data", linewidth=1.5)
plt.plot(time, damped_osc(time, *popt), "r--", linewidth=2,
         label="Fitted RLC Oscillator")
plt.xlabel("Time (frames)")
plt.ylabel("Activity level")
plt.title("Fig. 2 — Resonant LC/RLC fit to colony activity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Fig2_RLC_Fit.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Fig. 2 saved as Fig2_RLC_Fit.png")

# ====================== FIG. 3 — Quantum-inspired model ======================
omega = popt[1] if abs(popt[1]) > 0.01 else 0.12
H = omega * qt.sigmax() / 2
psi0 = qt.basis(2, 0)
tlist = np.linspace(0, len(time) * 0.1, 400)

result = qt.mesolve(H, psi0, tlist, [], [qt.sigmax(), qt.sigmay()])

plt.figure(figsize=(12, 5))
plt.plot(tlist, result.expect[0], "b-", label=r"$\langle\sigma_x\rangle$")
plt.plot(tlist, result.expect[1], "g--", label=r"$\langle\sigma_y\rangle$")
plt.xlabel("Scaled time")
plt.ylabel("Expectation value")
plt.title("Fig. 3 — Quantum-inspired model (Heisenberg picture)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Fig3_Quantum_Model.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Fig. 3 saved as Fig3_Quantum_Model.png")

print("\n🎉 ALL THREE FIGURES GENERATED AND SAVED!")
print("   Fig1_Empirical_Activity.png")
print("   Fig2_RLC_Fit.png")
print("   Fig3_Quantum_Model.png")
print("\nReady for the manuscript supplementary information!")
EOF

python generate_all_figures.py
