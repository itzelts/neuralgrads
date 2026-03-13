# Run a simulation of the Wilmes_2016 model

import os
import sys
import warnings
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["JAX_PLATFORMS"] = "cpu"
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

import numpy as np
import matplotlib.pyplot as plt
import jaxley as jx
from jaxley.channels import Leak
from jaxley.pumps import CaPump
from examples.Wilmes_2016.biophys.jaxley import (
    na3, na3dend, na3shifted,  # na3dend imported but not inserted (dendritic variant, no somatic gbar in config)
    kdrca1, kap,
    sca, it2, kca,
)

# ── Parameters from config_neuronmodel.py ─────────────────────────────────────

# Morphology (soma)
s_diam  = 18.5    # soma diameter (um)

# Passive
R_m     = 40000   # membrane resistance (Ω·cm²)
R_a     = 150     # axial resistivity (Ω·cm)
C_m     = 0.75    # capacitance (uF/cm²)
E_leak  = -70.0   # leak reversal potential (mV)

# Reversal potentials (mV)
E_Na    = 60.0
E_K     = -80.0
E_Ca    = 140.0

# Active conductances
g_Na             = 0.009   # na3         (S/cm²)
g_Na_ais_shifted = 0.3     # na3shifted  (S/cm²)
g_K              = 0.01    # kdrca1      (S/cm²)
g_KA             = 0.029   # kap         (S/cm²)
gsca             = 1.5e-4  # sca         (1.5 pS/um² → S/cm²)
git2             = 0.005   # it2         (S/cm²)
g_KCa            = 2.5e-4  # kca         (2.5 pS/um² → S/cm²)

temperature = 37.0

# ── Build cell ────────────────────────────────────────────────────────────────

cell = jx.Cell()
cell.set('length',            s_diam)
cell.set('radius',            s_diam / 2)
cell.set('axial_resistivity', R_a)
cell.set('capacitance',       C_m)

# ── Insert channels ───────────────────────────────────────────────────────────
# na3dend is a dendritic Na variant with vshift (dend_vshift=-5); no somatic gbar in config.

# somatic_channels = [na3(), na3shifted(), kdrca1(), kap(), sca(), it2(), kca()]
somatic_channels = [na3(), na3shifted(), kdrca1(), kap()]


for ch in somatic_channels:
    ch.set_tadj(temperature)
    cell.insert(ch)

cell.insert(CaPump())   # provides CaCon_i state required by kca
cell.insert(Leak())

# ── Set conductances ──────────────────────────────────────────────────────────
cell.set('gbar_na3',        g_Na)
cell.set('gbar_na3shifted', g_Na_ais_shifted)
cell.set('gbar_kdrca1',     g_K)
cell.set('gbar_kap',        g_KA)

cell.set('Leak_gLeak',      1.0 / R_m)
cell.set('Leak_eLeak',      E_leak)


# cell.set('gbar_na3',        g_Na)
# cell.set('gbar_na3shifted', g_Na_ais_shifted)
# cell.set('gbar_kdrca1',     g_K)
# cell.set('gbar_kap',        g_KA)
# cell.set('gbar_sca',        gsca)
# cell.set('gbar_it2',        git2)
# cell.set('gbar_kca',        g_KCa)
# cell.set('Leak_gLeak',      1.0 / R_m)
# cell.set('Leak_eLeak',      E_leak)

# ── Simulation ────────────────────────────────────────────────────────────────

current = jx.step_current(i_delay=100.0, i_dur=100.0, i_amp=0.1, delta_t=0.025, t_max=300.0)
cell.stimulate(current)
cell.record("v")

v = jx.integrate(cell, delta_t=0.025, t_max=300.0)
t = np.arange(0, 300, 0.025)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, v.T[:-1])
ax.set_ylim(-100, 60)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Voltage (mV)')
ax.set_title('Somatic Action Potentials')

os.makedirs('results', exist_ok=True)
fig.savefig('results/wilmes_2016_simulation.png')
