# Multicompartment Wilmes 2016 model in Jaxley
#
# Morphology mirrors neuronmodel.py (Wilmes et al. 2016):
#   soma → ais
#        → apical_prox → apical → tuft[0,1] → second_br[0-3]
#                      → oblique
#        → basal_main  → basal_br[0,1] → second_basal[0-3]
#
# Branch index map:
#   0=soma  1=ais  2=apical_prox  3=apical  4=oblique
#   5=basal_main  6=basal_br[0]  7=basal_br[1]
#   8-11=second_basal[0-3]  12-13=tuft[0-1]  14-17=second_br[0-3]
#
# Channel distribution (mirrors neuronmodel.py set_* methods):
#   All:       na3dend, kdrca1, kap (distance-scaled), sca, kca, CaPump, Leak
#   Non-soma:  also it2 (gbar=git2 in 500-750 um zone only, else 0)
#   AIS only:  also na3, na3shifted; gbar_na3dend overridden to 0

import os
import sys
import warnings
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["JAX_PLATFORMS"] = "cpu"
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jaxley as jx
from jaxley.connect import connect
from jaxley.channels import Leak
from jaxley.pumps import CaPump
from examples.Wilmes_2016.biophys.jaxley import (
    build_pre_cell, AMPA,
    na3, na3dend, na3shifted,
    kdrca1, kap,
    sca, it2, kca,
)

# parameters from config_neuronmodel.py

a_diam        = 2.0    # AIS diameter (um)
s_diam        = 18.5   # soma diameter (um)
d_diam        = 2.0    # dendrite diameter (um)
d_length      = 300.0  # dendrite length (um)
apical_length = 500.0  # apical total length (um)
slope_KA      = 5.0    # kap distance slope

R_m    = 40000  # membrane resistance (Ω·cm²)
R_a    = 150    # axial resistivity (Ω·cm)
C_m    = 0.75   # capacitance (uF/cm²)
E_leak = -70.0

g_Na             = 0.009   # na3dend everywhere (S/cm²)
g_Na_ais         = 0.3     # na3 on AIS (S/cm²)
g_Na_ais_shifted = 0.3     # na3shifted on AIS (S/cm²)
g_K              = 0.01    # kdrca1 (S/cm²)
g_KA             = 0.029   # kap base (S/cm²)
gsca             = 1.5e-4  # sca base (1.5 pS/um² → S/cm²)
git2             = 0.005   # it2 in 500-750 um zone (S/cm²)
g_KCa            = 2.5e-4  # kca (2.5 pS/um² → S/cm²)

temperature = 30.0  # celsius, from config_model_stim.py

# branch index constants

SOMA         = 0
AIS          = 1
APROX        = 2
APICAL       = 3
OBLIQUE      = 4
BASAL_MAIN   = 5
BASAL_BR     = [6, 7]
SECOND_BASAL = [8, 9, 10, 11]
TUFT         = [12, 13]
SECOND_BR    = [14, 15, 16, 17]

ALL_BRANCHES   = list(range(18))
NON_SOMA       = [i for i in ALL_BRANCHES if i != SOMA]

parents = [
    -1,   # 0: soma (root)
     0,   # 1: ais
     0,   # 2: apical_prox
     2,   # 3: apical
     2,   # 4: oblique
     0,   # 5: basal_main
     5,   # 6: basal_br[0]
     5,   # 7: basal_br[1]
     6,   # 8: second_basal[0]  → basal_br[0]
     7,   # 9: second_basal[1]  → basal_br[1]
     6,   # 10: second_basal[2] → basal_br[0]
     7,   # 11: second_basal[3] → basal_br[1]
     3,   # 12: tuft[0]
     3,   # 13: tuft[1]
    12,   # 14: second_br[0]   → tuft[0]
    13,   # 15: second_br[1]   → tuft[1]
    12,   # 16: second_br[2]   → tuft[0]
    13,   # 17: second_br[3]   → tuft[1]
]

# (total_length_um, diameter_um, ncomp)
branch_specs = {
    SOMA:       (s_diam,                  s_diam,                 1),
    AIS:        (a_diam,                  a_diam * 1.5,           1),
    APROX:      (apical_length / 5.0,     d_diam,                 4),
    APICAL:     (4 * apical_length / 5.0, d_diam,                 8),
    OBLIQUE:    (d_length,                d_diam / 5.0,           4),
    BASAL_MAIN: (d_length / 2.0,          d_diam / 2.0,           4),
}
for i in BASAL_BR:
    branch_specs[i] = (d_length / 2.0,  (2/3) * (d_diam / 2.0),    4)
for i in SECOND_BASAL:
    branch_specs[i] = (d_length / 2.0,  (2/3)**2 * (d_diam / 2.0), 4)
for i in TUFT:
    branch_specs[i] = (d_length,         (2/3) * d_diam,            4)
for i in SECOND_BR:
    branch_specs[i] = (d_length,         (2/3)**2 * d_diam,         4)

comp = jx.Compartment()
branches = [jx.Branch(comp, ncomp=branch_specs[i][2]) for i in range(18)]
cell = jx.Cell(branches, parents=parents)

for i, (total_length, diameter, ncomp) in branch_specs.items():
    cell.branch(i).set('length', total_length / ncomp)
    cell.branch(i).set('radius', diameter / 2.0)

cell.set('axial_resistivity', R_a)
cell.set('capacitance',       C_m)

# insert channels 
# all branches: na3dend, kdrca1, kap, sca, kca, CaPump, Leak
for ch_cls in [na3dend, kdrca1, kap, sca, kca]:
    ch = ch_cls()
    ch.set_tadj(temperature)
    cell.insert(ch)

cell.insert(CaPump())
cell.set("CaPump_gamma", 1.0) ## gamma is 0.5 defualt, we need 1.0 
cell.insert(Leak())

# non-soma branches: it2 (Ca-LVA; only active in 500-750 um zone)
it2_ch = it2()
it2_ch.set_tadj(temperature)
for i in NON_SOMA:
    cell.branch(i).insert(it2_ch)

# AIS only: na3 (high-density somatic Na) and na3shifted (shifted-threshold Na)
for ch_cls in [na3, na3shifted]:
    ch = ch_cls()
    ch.set_tadj(temperature)
    cell.branch(AIS).insert(ch)


# passive
cell.set('Leak_gLeak', 1.0 / R_m)
cell.set('Leak_eLeak', E_leak)

# na3dend: uniform except AIS where it is zeroed out
cell.set('gbar_na3dend', g_Na)
cell.branch(AIS).set('gbar_na3dend', 0.0)

# kdrca1: uniform
cell.set('gbar_kdrca1', g_K)

# per-compartment start distances from soma center
dist_start = {
    SOMA:       0,
    AIS:        s_diam / 2,
    APROX:      s_diam / 2,
    APICAL:     s_diam / 2 + apical_length / 5,
    OBLIQUE:    s_diam / 2 + apical_length / 5,
    BASAL_MAIN: s_diam / 2,
}
for i in BASAL_BR:
    dist_start[i] = s_diam / 2 + d_length / 2
for i in SECOND_BASAL:
    dist_start[i] = s_diam / 2 + d_length
for i in TUFT:
    dist_start[i] = s_diam / 2 + apical_length
for i in SECOND_BR:
    dist_start[i] = s_diam / 2 + apical_length + d_length

# kap: per-compartment distance-dependent scaling
cell.branch(SOMA).set('gbar_kap', g_KA)
for i in NON_SOMA:
    total_length, _, ncomp = branch_specs[i]
    comp_length = total_length / ncomp
    for j in range(ncomp):
        comp_dist = dist_start[i] + (j + 0.5) * comp_length
        dist = min(comp_dist, 500.0)
        gkap = g_KA * (1.0 + dist / (500.0 / slope_KA))
        cell.branch(i).comp(j).set('gbar_kap', gkap)

EXCLUDED_FROM_ZONE = [OBLIQUE] + BASAL_BR + SECOND_BASAL

# soma: sca × 2, kca × 2, no it2
cell.branch(SOMA).set('gbar_sca', gsca * 2)
cell.branch(SOMA).set('gbar_kca', g_KCa * 2)

# non-soma: per-compartment 500-750 um zone check
for i in NON_SOMA:
    total_length, _, ncomp = branch_specs[i]
    comp_length = total_length / ncomp
    for j in range(ncomp):
        comp_dist = dist_start[i] + (j + 0.5) * comp_length
        in_zone = (500 < comp_dist < 750) and (i not in EXCLUDED_FROM_ZONE)
        if in_zone:
            cell.branch(i).comp(j).set('gbar_it2', git2)
            cell.branch(i).comp(j).set('gbar_sca', gsca * 3)
            cell.branch(i).comp(j).set('gbar_kca', g_KCa)
        else:
            cell.branch(i).comp(j).set('gbar_it2', 0.0)
            cell.branch(i).comp(j).set('gbar_sca', gsca)
            cell.branch(i).comp(j).set('gbar_kca', g_KCa)

# AIS: high-density na3 and na3shifted
cell.branch(AIS).set('gbar_na3',        g_Na_ais)
cell.branch(AIS).set('gbar_na3shifted', g_Na_ais_shifted)
        
# from model_stim.py / config_model_stim.py:
# POST_AMP=0.3 nA, dur=2 ms, WARM_UP=1000 ms, delta_t(STDP)=-5 ms
# i_delay = WARM_UP + delta_t = 1000 - 5 = 995 ms
# total_time = WARM_UP + NO_REPS*(1000/freq) + 100 = 1000 + 25 + 100 = 1125 ms

delta_t = 0.1
t_max   = 1125.0
pre_i_delay = 995.0
pre_i_dur   = 0.5
pre_i_amp   = 0.1   # nA
g_ampa_uS   = 0.006 # 6 nS

pre_cell = build_pre_cell()
net = jx.Network([pre_cell, cell])
net.pumped_ions = list({p.ion_name for p in net.pumps})

connect(
    net.cell(0).branch(0).loc(0.5),
    net.cell(1).branch(SOMA).loc(0.5),
    AMPA(),
)
net.select(edges=0).set("AMPA_gAMPA", g_ampa_uS)

pre_current = jx.step_current(
    i_delay=pre_i_delay, i_dur=pre_i_dur, i_amp=pre_i_amp, delta_t=delta_t, t_max=t_max
)
net.cell(0).branch(0).loc(0.5).stimulate(pre_current)
net.cell(1).branch(SOMA).loc(0.5).record("v")
net.cell(1).branch(APICAL).loc(0.9).record("v")
net.cell(1).branch(OBLIQUE).loc(0.5).record("v")
net.cell(1).branch(BASAL_MAIN).loc(0.5).record("v")
net.cell(1).branch(APROX).loc(0.5).record("v")

v = jx.integrate(net, delta_t=delta_t, t_max=t_max)
t = np.arange(0, t_max, delta_t)

os.makedirs('results', exist_ok=True)


fig1, ax1 = plt.subplots(figsize=(10, 4))
v_arr = np.asarray(v)
t_plot = np.arange(v_arr.shape[1]) * delta_t
ax1.plot(t_plot, v_arr[0], color='black', label='Soma')
ax1.plot(t_plot, v_arr[1], color='dodgerblue', label='Apical')
ax1.plot(t_plot, v_arr[2], color='steelblue', label='Oblique')
ax1.plot(t_plot, v_arr[3], color='tomato', label='Basal')
ax1.plot(t_plot, v_arr[4], color='royalblue', label='Apical prox')
ax1.set_xlim(990, 1100)
ax1.set_ylim(-80, 40)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Voltage (mV)')
ax1.set_title('Voltage Traces — Pre (HH) → AMPA → Wilmes 2016 Multicompartment')
ax1.legend()
fig1.tight_layout()
fig1.savefig('results/wilmes_2016_network_traces.png', dpi=150)


cell.compute_xyz()
fig2, ax2 = plt.subplots(figsize=(6, 9))

region_colors = {
    SOMA:      'black',
    AIS:       'dimgray',
    APROX:     'royalblue',
    APICAL:    'dodgerblue',
    OBLIQUE:   'steelblue',
    BASAL_MAIN:'tomato',
}
for i in BASAL_BR:
    region_colors[i] = 'salmon'
for i in SECOND_BASAL:
    region_colors[i] = 'lightsalmon'
for i in TUFT:
    region_colors[i] = 'cornflowerblue'
for i in SECOND_BR:
    region_colors[i] = 'lightblue'

for i, color in region_colors.items():
    cell.branch(i).vis(ax=ax2, color=color)

ax2.set_title('Wilmes 2016 — Morphology')
ax2.set_xlabel('x (μm)')
ax2.set_ylabel('y (μm)')
ax2.legend(handles=[
    Line2D([0], [0], color='black',          lw=2, label='Soma'),
    Line2D([0], [0], color='dimgray',        lw=2, label='AIS'),
    Line2D([0], [0], color='royalblue',      lw=2, label='Apical prox / trunk'),
    Line2D([0], [0], color='cornflowerblue', lw=2, label='Apical tuft (sca×3, it2 active)'),
    Line2D([0], [0], color='steelblue',      lw=2, label='Oblique'),
    Line2D([0], [0], color='tomato',         lw=2, label='Basal main'),
    Line2D([0], [0], color='salmon',         lw=2, label='Basal branches'),
    Line2D([0], [0], color='lightsalmon',    lw=2, label='Basal 2nd order'),
    Line2D([0], [0], color='lightblue',      lw=2, label='Apical tuft 2nd order'),
], loc='upper right', fontsize=7)
fig2.tight_layout()
fig2.savefig('results/wilmes_network_morphology.png', dpi=150)

fig3, ax3 = plt.subplots(1, 1, figsize=(5, 10))
try:
    net.compute_xyz()
    # net.rotate(180)
    net.arrange_in_layers(
        layers=[1, 1],
        within_layer_offset=250,
        between_layer_offset=600,
    )
    net.vis(ax=ax3, detail="full")
except Exception:
    pre_cell.compute_xyz()
    cell.compute_xyz()
    pre_cell.vis(ax=ax3, color="gray")
    cell.vis(ax=ax3, color="black")
ax3.set_title("Pre/Post Network (AMPA onto soma)")
fig3.tight_layout()
fig3.savefig('results/wilmes_2016_network.png', dpi=150)