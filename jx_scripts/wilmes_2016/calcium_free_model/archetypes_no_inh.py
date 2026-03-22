"""
Detection windows:
Soma spike:
  Search from T_INIT (= T_EXC_ON − 5 ms) onwards.  The EPSP from the target
  synapse must first travel from the dendrite to the soma before it can
  contribute to firing
bAP at target:
  Anchor to the detected soma spike time T_spike and search the next
  BAP_WINDOW_MS (10 ms).  By T_spike the inbound EPSP at the target site
  is in its decay phase, so we are looking for the return bAP signal.
  10 ms covers even the most distal compartments (farthest needs like 10 ms
  in bap_time_heatmap.py)
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JAX_PLATFORMS"] = "cpu"
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import jaxley as jx
from jaxley.connect import connect

from examples.Wilmes_2016.biophys.jaxley import (
    build_wilmes_cell_no_ca,
    build_pre_cell,
    AMPA,
    SOMA, APROX, APICAL, OBLIQUE, BASAL_MAIN,
)
from examples.Wilmes_2016.biophys.jaxley.cells_no_ca import (
    branch_specs, dist_start,
    BASAL_BR, SECOND_BASAL, TUFT,
)
from utils import detect_first_spike_time, measure_pre_latency

dt = 0.1
t_max = 1125.0
T_EXC_ON = 995.0
PRE_STIM_DUR = 0.5
PRE_STIM_AMP = 0.1

G_SOMA_NS = 4.0
G_SOMA_US = G_SOMA_NS / 1000.0

G_TARGET_MIN_NS = 0.0
G_TARGET_MAX_NS = 25.0
N_G = 16
G_TARGET_NS = np.linspace(G_TARGET_MIN_NS, G_TARGET_MAX_NS, N_G)
G_TARGET_US = G_TARGET_NS / 1000.0

# Soma spike detection
SPIKE_CROSS = 0.0    # mV — zero-crossing threshold
SPIKE_PEAK = 20.0   # mV — minimum peak voltage to count as AP
REFRACTORY_MS = 2.0
# start search 5 ms before excitation so we catch spikes even when the EPSP
# from a near soma target synapse reaches and fires the soma very quickly
T_INIT = T_EXC_ON - 5.0

# bAP detection at target site
# search starts at T_spike (detected soma spike time); the bAP must travel from
# soma back to the target, so we look for BAP_WINDOW_MS ms after T_spike
BAP_DISTAL_THRESH = -20.0   # mV — peak at target must exceed this
# bap_time_heatmap shows warm colours (> -20 mV) return to baseline within
# 6-7 ms even at the farthest apical compartments
BAP_WINDOW_MS = 10.0 # ms  — covers the full bAP depolarisation window


def _comps_on_branch(br: int) -> list[tuple[int, int]]:
    return [(br, c) for c in range(branch_specs[br][2])]


def build_paths() -> list[tuple[str, list[tuple[int, int]]]]:
    apical = (
        "Apical trunk  (APROX → APICAL → TUFT)",
        _comps_on_branch(APROX)
        + _comps_on_branch(APICAL)
        + _comps_on_branch(TUFT[0]),
    )
    oblique = (
        "Oblique branch",
        _comps_on_branch(OBLIQUE),
    )
    basal = (
        "Basal path  (BASAL_MAIN → BASAL_BR → 2nd_BASAL)",
        _comps_on_branch(BASAL_MAIN)
        + _comps_on_branch(BASAL_BR[0])
        + _comps_on_branch(SECOND_BASAL[0]),
    )
    return [apical, oblique, basal]


def _comp_loc(branch_id: int, comp_id: int) -> float:
    return (comp_id + 0.5) / branch_specs[branch_id][2]


def _comp_dist_um(branch_id: int, comp_id: int) -> float:
    total, _, ncomp = branch_specs[branch_id]
    return dist_start[branch_id] + (comp_id + 0.5) * (total / ncomp)



def build_network(
    target_branch: int,
    target_loc: float,
    g_target_uS_init: float,
    t_pre_current: float,
) -> jx.Network:
    pre_soma = build_pre_cell()
    pre_target = build_pre_cell()
    post = build_wilmes_cell_no_ca()

    net = jx.Network([pre_soma, pre_target, post])
    net.pumped_ions = list({p.ion_name for p in net.pumps})

    connect(
        net.cell(0).branch(0).loc(0.5),
        net.cell(2).branch(SOMA).loc(0.5),
        AMPA(),
    )
    net.select(edges=0).set("AMPA_gAMPA", G_SOMA_US)

    connect(
        net.cell(1).branch(0).loc(0.5),
        net.cell(2).branch(target_branch).loc(target_loc),
        AMPA(),
    )
    net.select(edges=1).set("AMPA_gAMPA", g_target_uS_init)

    stim = jx.step_current(
        i_delay=t_pre_current, i_dur=PRE_STIM_DUR, i_amp=PRE_STIM_AMP,
        delta_t=dt, t_max=t_max,
    )
    net.cell(0).branch(0).loc(0.5).stimulate(stim)
    net.cell(1).branch(0).loc(0.5).stimulate(stim)

    net.cell(2).branch(SOMA).loc(0.5).record("v")
    net.cell(2).branch(target_branch).loc(target_loc).record("v")
    return net



def run_sweep(
    path: list[tuple[int, int]],
    hh_latency: float,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_d = len(path)
    n_g = len(G_TARGET_US)
    soma_spiked   = np.zeros((n_d, n_g), dtype=bool)
    bap_at_target = np.zeros((n_d, n_g), dtype=bool)

    t_pre = T_EXC_ON - hh_latency
    g_batch = jnp.asarray(G_TARGET_US)
    bap_steps = int(BAP_WINDOW_MS / dt)

    for i_d, (branch_id, comp_id) in enumerate(path):
        loc = _comp_loc(branch_id, comp_id)
        dist_um = _comp_dist_um(branch_id, comp_id)

        net = build_network(
            target_branch=branch_id,
            target_loc=loc,
            g_target_uS_init=float(G_TARGET_US[0]),
            t_pre_current=t_pre,
        )

        # vmap over conductance axis
        def simulate_one(g_uS):
            ps = net.select(edges=1).data_set("AMPA_gAMPA", g_uS, param_state=None)
            return jx.integrate(net, param_state=ps, delta_t=dt, t_max=t_max)

        t0  = time.time()
        out = jax.jit(jax.vmap(simulate_one))(g_batch)
        out.block_until_ready()
        print(f"  row {i_d + 1}/{n_d}  dist={dist_um:.0f} µm  ({time.time() - t0:.1f} s)")

        v_soma = np.asarray(out[:, 0, :-1])   # (n_g, T)
        v_target = np.asarray(out[:, 1, :-1])   # (n_g, T)

        for i_g in range(n_g):
            t_spike = detect_first_spike_time(v_soma[i_g], t, T_INIT)
            if t_spike is None:
                continue
            soma_spiked[i_d, i_g] = True

            idx0 = int(np.searchsorted(t, t_spike))
            idx1 = min(len(t), idx0 + bap_steps)
            bap_at_target[i_d, i_g] = (
                float(np.max(v_target[i_g, idx0:idx1])) > BAP_DISTAL_THRESH
            )
    return soma_spiked, bap_at_target



# colormap: black=AP−, green=AP+/bAP+, red=AP+/bAP−
_STATE_CMAP = mcolors.ListedColormap(["black", "#e74c3c", "#2ecc71"])
# encode states: 0=AP−, 1=AP+/bAP−, 2=AP+/bAP+
_STATE_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _STATE_CMAP.N)


def _state_matrix(soma_spiked: np.ndarray, bap_at_target: np.ndarray) -> np.ndarray:
    """Build integer (n_d, n_g) state array: 0=AP−, 1=AP+bAP−, 2=AP+bAP+."""
    s = np.zeros(soma_spiked.shape, dtype=int)
    s[soma_spiked & ~bap_at_target] = 1
    s[soma_spiked &  bap_at_target] = 2
    return s


def _pcolor_edges(centers: np.ndarray) -> np.ndarray:
    if len(centers) == 1:
        return np.array([centers[0] - 5.0, centers[0] + 5.0])
    d = np.diff(centers)
    e = np.empty(len(centers) + 1)
    e[1:-1] = centers[:-1] + d / 2
    e[0] = centers[0]  - d[0]  / 2
    e[-1] = centers[-1] + d[-1] / 2
    return e


def plot_path_heatmap(
    soma_spiked: np.ndarray,
    bap_at_target: np.ndarray,
    path: list[tuple[int, int]],
    path_name: str,
    out_path: str,
) -> None:
    n_d, n_g = soma_spiked.shape
    state    = _state_matrix(soma_spiked, bap_at_target)   # (n_d, n_g)

    dist_um = np.array([_comp_dist_um(b, c) for b, c in path])
    x_edges = _pcolor_edges(dist_um)
    y_edges = _pcolor_edges(G_TARGET_NS)

    fig, ax = plt.subplots(figsize=(max(7.5, n_d * 0.55 + 2.5), 5.5))

    ax.pcolormesh(
        x_edges, y_edges,
        state.T,           # (n_g, n_d) — rows=y, cols=x
        cmap=_STATE_CMAP, norm=_STATE_NORM,
        shading="flat",
    )

    # horizontal reference at the fixed soma synapse conductance
    ax.axhline(
        G_SOMA_NS, color="cyan", lw=0.9, ls="--", alpha=0.8,
        label=f"Soma synapse = {G_SOMA_NS:.0f} nS (fixed)",
    )

    ax.set_xlabel("Target synapse distance from soma (µm)", fontsize=10)
    ax.set_ylabel("Target synapse conductance (nS)", fontsize=10)
    ax.set_title(
        f"{path_name}\n"
        f"Fixed soma AMPA = {G_SOMA_NS:.0f} nS\n"
        f"BAP threshold = {BAP_DISTAL_THRESH} mV with BAP window = {BAP_WINDOW_MS:.0f} ms after soma spike",
        fontsize=9,
    )

    legend_elements = [
        mpatches.Patch(facecolor="black", label="AP− (soma did not fire)"),
        mpatches.Patch(facecolor="#2ecc71", label="AP+, bAP+ at target"),
        mpatches.Patch(facecolor="#e74c3c", label="AP+, bAP− at target (did not reach/attenuated)"),
        plt.Line2D([0], [0], color="cyan", lw=1.2, ls="--",
                   label=f"Soma synapse = {G_SOMA_NS:.0f} nS"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.85)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

def main():
    os.makedirs(os.path.join("results", "calcium_free_model", "archetypes_no_inh"), exist_ok=True)
    t = np.arange(0, t_max, dt)

    hh_latency = measure_pre_latency(build_pre_cell, T_EXC_ON, dt, t_max)
    print(f"Latency = {hh_latency:.2f} ms\n")

    paths = build_paths()

    for path_label, path_comps in paths:
        safe = path_label.split()[0].lower()
        print(
            f"Sweep: {path_label}\n"
            f"({len(path_comps)} locations × {N_G} conductances)\n"
        )
        soma_spiked, bap = run_sweep(path_comps, hh_latency, t)

        n_ap = int(soma_spiked.sum())
        n_bap = int(bap.sum())
        print(f"AP+: {n_ap}/{soma_spiked.size}, bAP+: {n_bap}/{soma_spiked.size}")

        out_path = os.path.join("results", "calcium_free_model", "archetypes_no_inh", f"archetypes_no_inh_{safe}.png")

        plot_path_heatmap(soma_spiked, bap, path_comps, path_label, out_path)
        print()


if __name__ == "__main__":
    main()
