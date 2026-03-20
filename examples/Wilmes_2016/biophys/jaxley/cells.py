"""
cell builders for the Wilmes 2016 Jaxley model
  from examples.Wilmes_2016.biophys.jaxley import (
      build_wilmes_cell
      build_pre_cell,
      SOMA, AIS, APICAL, TUFT, APROX, OBLIQUE, BASAL_MAIN,
      BASAL_BR, SECOND_BASAL, SECOND_BR, ALL_BRANCHES, NON_SOMA,
  )
  cell = build_wilmes_cell()
  pre = build_pre_cell()   # HH single-compartment presynaptic spike source
"""

import jaxley as jx
from jaxley.channels import Leak, HH, Na, K
from jaxley.pumps import CaPump

from . import (
    na3,
    na3dend,
    na3shifted,
    kdrca1,
    kap,
    sca,
    it2,
    kca,
)

a_diam = 2.0
s_diam = 18.5
d_diam = 2.0
d_length = 300.0
apical_length = 500.0
slope_KA = 5.0

R_m = 40000
R_a = 150
C_m = 0.75
E_leak = -70.0

g_Na = 0.009
g_Na_ais = 0.3
g_Na_ais_shifted = 0.3
g_K = 0.01
g_KA = 0.029
gsca = 1.5e-4
git2 = 0.005
g_KCa = 2.5e-4

temperature = 30.0

SOMA = 0
AIS = 1
APROX = 2
APICAL = 3
OBLIQUE = 4
BASAL_MAIN = 5
BASAL_BR = [6, 7]
SECOND_BASAL = [8, 9, 10, 11]
TUFT = [12, 13]
SECOND_BR = [14, 15, 16, 17]

ALL_BRANCHES = list(range(18))
NON_SOMA = [i for i in ALL_BRANCHES if i != SOMA]

parents = [
    -1, 0, 0, 2, 2, 0, 5, 5,
    6, 7, 6, 7, 3, 3, 12, 13, 12, 13,
]

branch_specs = {
    SOMA: (s_diam, s_diam, 1),
    AIS: (a_diam, a_diam * 1.5, 1),
    APROX: (apical_length / 5.0, d_diam, 4),
    APICAL: (4 * apical_length / 5.0, d_diam, 8),
    OBLIQUE: (d_length, d_diam / 5.0, 4),
    BASAL_MAIN: (d_length / 2.0, d_diam / 2.0, 4),
}
for i in BASAL_BR:
    branch_specs[i] = (d_length / 2.0, (2 / 3) * (d_diam / 2.0), 4)
for i in SECOND_BASAL:
    branch_specs[i] = (d_length / 2.0, (2 / 3) ** 2 * (d_diam / 2.0), 4)
for i in TUFT:
    branch_specs[i] = (d_length, (2 / 3) * d_diam, 4)
for i in SECOND_BR:
    branch_specs[i] = (d_length, (2 / 3) ** 2 * d_diam, 4)

EXCLUDED_FROM_ZONE = [OBLIQUE] + BASAL_BR + SECOND_BASAL

dist_start = {
    SOMA: 0,
    AIS: s_diam / 2,
    APROX: s_diam / 2,
    APICAL: s_diam / 2 + apical_length / 5,
    OBLIQUE: s_diam / 2 + apical_length / 5,
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


def build_wilmes_cell():
    comp = jx.Compartment()
    branches = [jx.Branch(comp, ncomp=branch_specs[i][2]) for i in range(18)]
    cell = jx.Cell(branches, parents=parents)

    for i, (total_length, diameter, ncomp) in branch_specs.items():
        cell.branch(i).set("length", total_length / ncomp)
        cell.branch(i).set("radius", diameter / 2.0)

    cell.set("axial_resistivity", R_a)
    cell.set("capacitance", C_m)

    for ch_cls in [na3dend, kdrca1, kap, sca, kca]:
        ch = ch_cls()
        ch.set_tadj(temperature)
        cell.insert(ch)

    cell.insert(CaPump())
    cell.set("CaPump_gamma", 1.0)
    cell.insert(Leak())

    it2_ch = it2()
    it2_ch.set_tadj(temperature)
    for i in NON_SOMA:
        cell.branch(i).insert(it2_ch)

    for ch_cls in [na3, na3shifted]:
        ch = ch_cls()
        ch.set_tadj(temperature)
        cell.branch(AIS).insert(ch)

    cell.set("Leak_gLeak", 1.0 / R_m)
    cell.set("Leak_eLeak", E_leak)
    cell.set("gbar_na3dend", g_Na)
    cell.branch(AIS).set("gbar_na3dend", 0.0)
    cell.set("gbar_kdrca1", g_K)

    cell.branch(SOMA).set("gbar_kap", g_KA)
    for i in NON_SOMA:
        total_length, _, ncomp = branch_specs[i]
        comp_length = total_length / ncomp
        for j in range(ncomp):
            comp_dist = dist_start[i] + (j + 0.5) * comp_length
            dist = min(comp_dist, 500.0)
            gkap = g_KA * (1.0 + dist / (500.0 / slope_KA))
            cell.branch(i).comp(j).set("gbar_kap", gkap)

    cell.branch(SOMA).set("gbar_sca", gsca * 2)
    cell.branch(SOMA).set("gbar_kca", g_KCa * 2)

    for i in NON_SOMA:
        total_length, _, ncomp = branch_specs[i]
        comp_length = total_length / ncomp
        for j in range(ncomp):
            comp_dist = dist_start[i] + (j + 0.5) * comp_length
            in_zone = (500 < comp_dist < 750) and (i not in EXCLUDED_FROM_ZONE)
            if in_zone:
                cell.branch(i).comp(j).set("gbar_it2", git2)
                cell.branch(i).comp(j).set("gbar_sca", gsca * 3)
                cell.branch(i).comp(j).set("gbar_kca", g_KCa)
            else:
                cell.branch(i).comp(j).set("gbar_it2", 0.0)
                cell.branch(i).comp(j).set("gbar_sca", gsca)
                cell.branch(i).comp(j).set("gbar_kca", g_KCa)

    cell.branch(AIS).set("gbar_na3", g_Na_ais)
    cell.branch(AIS).set("gbar_na3shifted", g_Na_ais_shifted)

    return cell


def build_pre_cell():
    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    cell.insert(HH())
    return cell
