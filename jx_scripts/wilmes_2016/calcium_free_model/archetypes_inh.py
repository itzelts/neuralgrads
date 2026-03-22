import os
import sys
import time
from typing import Any
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
    AMPA, GABAa,
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

INH_BRANCH = APROX
INH_LOC = 0.9
G_INH_NS = 60.0
G_INH_US = G_INH_NS / 1000.0
E_GABA = -73.0 # mV
INH_DELAY_MS = -0.5 # ms rel to soma spike

# distance of the inhibitory synapse from soma (for plot annotation)
INH_DIST_UM = dist_start[INH_BRANCH] + INH_LOC * branch_specs[INH_BRANCH][0]

# soma spike detection
SPIKE_CROSS = 0.0
SPIKE_PEAK = 20.0
REFRACTORY_MS = 2.0
T_INIT = T_EXC_ON - 5.0

# bAP detection
BAP_DISTAL_THRESH = -20.0
BAP_WINDOW_MS = 10.0

RESULTS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    "results", "synapse_location_weight_heatmap_inh",
)



def _comps_on_branch(br: int) -> list[tuple[int, int]]:
    return [(br, c) for c in range(branch_specs[br][2])]


def build_paths() -> list[tuple[str, list[tuple[int, int]], bool]]:
    apical = (
        "Apical trunk  (APROX → APICAL → TUFT)",
        _comps_on_branch(APROX)
        + _comps_on_branch(APICAL)
        + _comps_on_branch(TUFT[0]),
        True,   # APROX is in this path
    )
    oblique = (
        "Oblique branch",
        _comps_on_branch(OBLIQUE),
        True,   # bAP from soma passes through APROX 0.9 to reach OBLIQUE
    )
    basal = (
        "Basal path  (BASAL_MAIN → BASAL_BR → 2nd_BASAL)",
        _comps_on_branch(BASAL_MAIN)
        + _comps_on_branch(BASAL_BR[0])
        + _comps_on_branch(SECOND_BASAL[0]),
        False,  # BASAL branches off SOMA, bypasses APROX entirely
    )
    return [apical, oblique, basal]


def _comp_loc(branch_id: int, comp_id: int) -> float:
    return (comp_id + 0.5) / branch_specs[branch_id][2]


def _comp_dist_um(branch_id: int, comp_id: int) -> float:
    total, _, ncomp = branch_specs[branch_id]
    return dist_start[branch_id] + (comp_id + 0.5) * (total / ncomp)



def build_network_no_inh(
    target_branch: int,
    target_loc: float,
    g_target_uS_init: float,
    t_pre_current: float,
) -> jx.Network:
    pre_soma = build_pre_cell()
    pre_target = build_pre_cell()
    post = build_wilmes_cell_no_ca()

    net = jx.Network([pre_soma, pre_target, post])

    connect(net.cell(0).branch(0).loc(0.5),
            net.cell(2).branch(SOMA).loc(0.5), AMPA())
    net.select(edges=0).set("AMPA_gAMPA", G_SOMA_US)

    connect(net.cell(1).branch(0).loc(0.5),
            net.cell(2).branch(target_branch).loc(target_loc), AMPA())
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


def build_network_with_inh(
    target_branch: int,
    target_loc: float,
    g_target_uS_init: float,
    t_pre_current: float,
) -> jx.Network:
    pre_soma = build_pre_cell()
    pre_target = build_pre_cell()
    pre_inh= build_pre_cell()
    post = build_wilmes_cell_no_ca()

    net = jx.Network([pre_soma, pre_target, pre_inh, post])

    connect(net.cell(0).branch(0).loc(0.5),
            net.cell(3).branch(SOMA).loc(0.5), AMPA())
    net.select(edges=0).set("AMPA_gAMPA", G_SOMA_US)

    connect(net.cell(1).branch(0).loc(0.5),
            net.cell(3).branch(target_branch).loc(target_loc), AMPA())
    net.select(edges=1).set("AMPA_gAMPA", g_target_uS_init)

    connect(net.cell(2).branch(0).loc(0.5),
            net.cell(3).branch(INH_BRANCH).loc(INH_LOC), GABAa())
    net.select(edges=2).set("GABAa_gGABAa", G_INH_US)
    net.select(edges=2).set("GABAa_eGABAa", E_GABA)

    stim_exc = jx.step_current(
        i_delay=t_pre_current, i_dur=PRE_STIM_DUR, i_amp=PRE_STIM_AMP,
        delta_t=dt, t_max=t_max,
    )
    net.cell(0).branch(0).loc(0.5).stimulate(stim_exc)
    net.cell(1).branch(0).loc(0.5).stimulate(stim_exc)
    # pre_inh stimulus injected via data_stimulate at vmap time (not baked in)

    net.cell(3).branch(SOMA).loc(0.5).record("v")
    net.cell(3).branch(target_branch).loc(target_loc).record("v")
    return net


def run_sweep_with_inh(
    path: list[tuple[int, int]],
    hh_latency: float,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_d = len(path)
    n_g = len(G_TARGET_US)
    soma_spiked   = np.zeros((n_d, n_g), dtype=bool)
    bap_at_target = np.zeros((n_d, n_g), dtype=bool)

    t_pre      = T_EXC_ON - hh_latency
    g_batch    = jnp.asarray(G_TARGET_US)
    bap_steps  = int(BAP_WINDOW_MS / dt)
    # dummy inh timing for non-spiking cells — fires well after t_max so it
    # never actually reaches the post-cell
    T_DUMMY_INH = t_max + 1000.0

    for i_d, (branch_id, comp_id) in enumerate(path):
        loc     = _comp_loc(branch_id, comp_id)
        dist_um = _comp_dist_um(branch_id, comp_id)
        t0      = time.time()

        # pass 1: no inhibition, vmap over conductances
        net1 = build_network_no_inh(branch_id, loc, float(G_TARGET_US[0]), t_pre)

        def sim_pass1(g_uS):
            ps = net1.select(edges=1).data_set("AMPA_gAMPA", g_uS, param_state=None)
            return jx.integrate(net1, param_state=ps, delta_t=dt, t_max=t_max)

        out1 = jax.jit(jax.vmap(sim_pass1))(g_batch)
        out1.block_until_ready()
        v_soma1 = np.asarray(out1[:, 0, :-1])   # (n_g, T)

        # detect spike times from pass 1
        t_spike_vals = []
        for i_g in range(n_g):
            ts = detect_first_spike_time(v_soma1[i_g], t, T_INIT)
            t_spike_vals.append(ts)
            if ts is not None:
                soma_spiked[i_d, i_g] = True

        # pass 2: with inhibition, vmap over (conductance, inh_stim)
        net2 = build_network_with_inh(branch_id, loc, float(G_TARGET_US[0]), t_pre)

        # build per conductance inh stimulus array: (n_g, T)
        # cells that spiked get inh at t_spike + delay; others get a dummy
        stim_inh_arrays = jnp.stack([
            jnp.array(jx.step_current(
                i_delay=(t_spike_vals[i_g] + INH_DELAY_MS - hh_latency)
                        if soma_spiked[i_d, i_g] else T_DUMMY_INH,
                i_dur=PRE_STIM_DUR, i_amp=PRE_STIM_AMP,
                delta_t=dt, t_max=t_max,
            ))
            for i_g in range(n_g)
        ])  # (n_g, T)

        def sim_pass2(g_uS, stim_inh):
            ps        = net2.select(edges=1).data_set("AMPA_gAMPA", g_uS, param_state=None)
            data_stim = net2.cell(2).branch(0).loc(0.5).data_stimulate(stim_inh)
            return jx.integrate(net2, param_state=ps, data_stimuli=data_stim,
                                 delta_t=dt, t_max=t_max)

        out2 = jax.jit(jax.vmap(sim_pass2))(g_batch, stim_inh_arrays)
        out2.block_until_ready()
        v_target2 = np.asarray(out2[:, 1, :-1])   # (n_g, T)

        print(f"  row {i_d + 1}/{n_d}  dist={dist_um:.0f} µm  ({time.time() - t0:.1f} s)")

        # detect bAP using soma spike time from pass 1 (inh is post-spike so
        # soma_spiked is the same in both passes)
        for i_g in range(n_g):
            if not soma_spiked[i_d, i_g]:
                continue
            ts   = t_spike_vals[i_g]
            idx0 = int(np.searchsorted(t, ts))
            idx1 = min(len(t), idx0 + bap_steps)
            bap_at_target[i_d, i_g] = (
                float(np.max(v_target2[i_g, idx0:idx1])) > BAP_DISTAL_THRESH
            )

    return soma_spiked, bap_at_target


_STATE_CMAP = mcolors.ListedColormap(["black", "#e74c3c", "#2ecc71"])
_STATE_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _STATE_CMAP.N)


def _state_matrix(soma_spiked: np.ndarray, bap_at_target: np.ndarray) -> np.ndarray:
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
    e[0]    = centers[0]  - d[0]  / 2
    e[-1]   = centers[-1] + d[-1] / 2
    return e


def plot_path_heatmap(
    soma_spiked: np.ndarray,
    bap_at_target: np.ndarray,
    path: list[tuple[int, int]],
    path_name: str,
    inh_on_path: bool,
    out_path: str,
) -> None:
    n_d, n_g = soma_spiked.shape
    state    = _state_matrix(soma_spiked, bap_at_target)

    dist_um = np.array([_comp_dist_um(b, c) for b, c in path])
    x_edges = _pcolor_edges(dist_um)
    y_edges = _pcolor_edges(G_TARGET_NS)
    x_min, x_max = x_edges[0], x_edges[-1]

    fig, ax = plt.subplots(figsize=(max(7.5, n_d * 0.55 + 2.5), 5.5))

    ax.pcolormesh(
        x_edges, y_edges,
        state.T,
        cmap=_STATE_CMAP, norm=_STATE_NORM,
        shading="flat",
    )

    # soma synapse reference line
    ax.axhline(G_SOMA_NS, color="cyan", lw=0.9, ls="--", alpha=0.8)

    # inhibition annotation
    if inh_on_path and x_min <= INH_DIST_UM <= x_max:
        # inh location falls inside the plot range (apical path)
        ax.axvline(INH_DIST_UM, color="magenta", lw=1.4, ls="-.",
                   label=f"GABAa inh ({INH_DIST_UM:.0f} µm, {G_INH_NS:.0f} nS, "
                         f"+{INH_DELAY_MS} ms after spike)")
    elif inh_on_path and INH_DIST_UM < x_min:
        # inh is proximal to the plot range (oblique path)
        ax.annotate(
            f"GABAa inh at {INH_DIST_UM:.0f} µm (proximal)\n"
            f"All compartments are distal to inhibition",
            xy=(x_min, G_TARGET_NS[-1]),
            xytext=(x_min + (x_max - x_min) * 0.02, G_TARGET_NS[-1] * 0.88),
            fontsize=7.5, color="magenta",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="magenta", alpha=0.8),
        )
    else:
        # basal path — inh not on path
        ax.text(
            0.98, 0.97,
            f"Inh at APROX {INH_LOC} ({INH_DIST_UM:.0f} µm)\nnot on basal path",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="magenta",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="magenta", alpha=0.8),
        )

    ax.set_xlabel("Target synapse distance from soma (µm)", fontsize=10)
    ax.set_ylabel("Target synapse conductance (nS)", fontsize=10)
    ax.set_title(
        f"{path_name}\n"
        f"Fixed soma AMPA = {G_SOMA_NS:.0f} nS\n"
        f"BAP threshold = {BAP_DISTAL_THRESH} mV with BAP window = {BAP_WINDOW_MS:.0f} ms after soma spike\n"
        f"GABAa Inhibition @ APROX {INH_LOC} = {G_INH_NS:.0f} nS, {INH_DELAY_MS} ms after spike",
        fontsize=9,
    )

    legend_elements = [
        mpatches.Patch(facecolor="black",label="AP− (soma did not fire)"),
        mpatches.Patch(facecolor="#2ecc71", label="AP+, bAP+ at target"),
        mpatches.Patch(facecolor="#e74c3c", label="AP+, bAP− at target (did not reach/attenuated)"),
        plt.Line2D([0], [0], color="cyan", lw=1.2, ls="--",
                   label=f"Soma synapse = {G_SOMA_NS:.0f} nS"),
        plt.Line2D([0], [0], color="magenta", lw=1.4, ls="-.",
                   label=f"GABAa inh ({INH_DIST_UM:.0f} µm)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.85)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

def main():
    os.makedirs(os.path.join("results", "calcium_free_model", "archetypes_inh"), exist_ok=True)
    t = np.arange(0, t_max, dt)

    hh_latency = measure_pre_latency(build_pre_cell, T_EXC_ON, dt, t_max)
    print(f"Latency = {hh_latency:.2f} ms")
    print(f"Inhibition: APROX loc {INH_LOC}, ({INH_DIST_UM:.1f} µm from soma)  "
          f"{G_INH_NS:.0f} nS GABAa {INH_DELAY_MS} ms after spike\n")

    paths = build_paths()

    for path_label, path_comps, inh_on_path in paths:
        safe = path_label.split()[0].lower()
        print(
            f"Sweep: {path_label}\n"
            f"({len(path_comps)} locations × {N_G} conductances)\n"
            f"inh_on_path={inh_on_path}"
        )
        soma_spiked, bap = run_sweep_with_inh(path_comps, hh_latency, t)

        n_ap  = int(soma_spiked.sum())
        n_bap = int(bap.sum())
        print(f"AP+: {n_ap}/{soma_spiked.size}, bAP+: {n_bap}/{soma_spiked.size}")

        out_path = os.path.join("results", "calcium_free_model", "archetypes_inh", f"archetypes_inh_{safe}.png")
        plot_path_heatmap(soma_spiked, bap, path_comps, path_label, inh_on_path, out_path)
        print()


if __name__ == "__main__":
    main()
