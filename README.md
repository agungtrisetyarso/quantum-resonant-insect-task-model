# quantum-resonant-insect-task-model
Quantum-inspired resonant LC-circuit + response-threshold model of task allocation in Ooceraea biroi colonies (extension of QICIF framework)
# Quantum-Inspired Resonant Dynamics of Task Substitution in Eusocial Insect Colonies

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![QuTiP](https://img.shields.io/badge/QuTiP-5+-green.svg)](https://qutip.org/)

**Extension of the Quantum-Inspired Computational Intelligence Framework (QICIF) to biological systems**

This repository implements a **hybrid response-threshold + resonant LC-circuit + quantum-inspired model** of dynamic task allocation in eusocial insects, using high-resolution colony tracking data from *Ooceraea biroi*.

It directly bridges the original QICIF automation manuscript (oscillatory labor-capital substitution via LC circuits and entangled states) with real biological data from Ulrich et al. (2021).

---

### Abstract
Division of labor in eusocial insects emerges from simple individual response thresholds yet produces robust oscillatory reallocation under perturbation. We extend classical response-threshold models by mapping them onto a resonant LC-circuit analogy (workers as capacitors, task stimuli as inductors) and a quantum-inspired formalism (entangled colony states evolved in the Heisenberg picture using QuTiP). The hybrid framework quantitatively reproduces empirical damped oscillations observed in *Ooceraea biroi* colonies and reveals striking parallels to automation systems.

---

### Key Features
- ✅ Loads and cleans real *Ooceraea biroi* tracking data (Ulrich et al., 2021)
- ✅ Detects oscillatory peaks in colony activity
- ✅ Fits damped resonant LC/RLC oscillator (Fig. 2)
- ✅ Runs QuTiP quantum simulation (Heisenberg picture, entangled states) (Fig. 3)
- ✅ One-click script that generates all publication-ready figures
- ✅ Fully reproducible Python pipeline (conda environment)

---

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/quantum-resonant-insect-task-model.git
cd quantum-resonant-insect-task-model

# 2. Create and activate conda environment
conda env create -f environment.yml   # (or manually as below)
conda activate insect-model

# Manual installation if you prefer:
conda create -n insect-model python=3.11
conda activate insect-model
conda install -c conda-forge numpy pandas matplotlib scipy qutip
