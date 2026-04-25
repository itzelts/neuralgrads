import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JAX_PLATFORMS"] = "cpu"
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

import numpy as np
import matplotlib.pyplot as plt
import jaxley as jx
from jaxley.connect import connect

from examples.Wilmes_2016.biophys.jaxley import (
    build_wilmes_cell,
    build_pre_cell,
    AMPA,
    SOMA, APROX, APICAL, OBLIQUE, BASAL_MAIN,
)
from examples.Wilmes_2016.biophys.jaxley.cells import (
    branch_specs, dist_start,
)
from utils import measure_pre_latency

dt = 0.1
t_max = 1125.0

T_EXC_ON = 995.0
PRE_STIM_DUR = 0.5
PRE_STIM_AMP = 0.1

G_SOMA_AMPA = 0.006 # uS = 7 nS — threshold excitation
SOMA_EXC_LOC = 0.5
SOMA_SPIKE_THRESH = 0.0 # mV

# window around soma spike to display
PRE_SPIKE_MS  = 5.0
POST_SPIKE_MS = 30.0


def _comps_on_branch(br: int) -> list[tuple[int, int, float]]:
    """Return [(branch, comp_idx, distance_from_soma_um)] for every comp on br."""
    length, _, ncomp = branch_specs[br]
    comp_len = length / ncomp
    ds = dist_start[br]
    return [(br, c, ds + (c + 0.5) * comp_len) for c in range(ncomp)]


def build_paths() -> list[tuple[str, list[tuple[int, int, float]]]]:
    """
    Returns list of (label, [(branch_idx, comp_idx, distance_um), ...]).
    Compartments are ordered from soma outward along each path.
    """
    apical_trunk = (
        "Apical trunk  (SOMA → APROX → APICAL → TUFT)",
        _comps_on_branch(SOMA)
        + _comps_on_branch(APROX)
        + _comps_on_branch(APICAL)
        + _comps_on_branch(12),   # TUFT[0] = branch 12
    )
    oblique = (
        "Oblique branch  (from APROX junction)",
        _comps_on_branch(OBLIQUE),
    )
    basal = (
        "Basal path  (BASAL_MAIN → BASAL_BR → 2nd_BASAL)",
        _comps_on_branch(BASAL_MAIN)
        + _comps_on_branch(6)    # BASAL_BR[0]
        + _comps_on_branch(8),   # SECOND_BASAL[0]
    )
    return [apical_trunk, oblique, basal]


def run_simulation(
    hh_latency: float,
    paths: list[tuple[str, list[tuple[int, int, float]]]],
) -> tuple[np.ndarray, list[np.ndarray], float]:
    pre_exc = build_pre_cell()
    post = build_wilmes_cell()
    net = jx.Network([pre_exc, post])
    net.pumped_ions = list({p.ion_name for p in net.pumps})
    connect(
        net.cell(0).branch(0).loc(0.5),
        net.cell(1).branch(SOMA).loc(SOMA_EXC_LOC),
        AMPA(),
    )
    net.select(edges=0).set("AMPA_gAMPA", G_SOMA_AMPA)

    stim = jx.step_current(
        i_delay=T_EXC_ON - hh_latency, 
        i_dur=PRE_STIM_DUR, 
        i_amp=PRE_STIM_AMP,
        delta_t=dt, t_max=t_max,
    )
    net.cell(0).branch(0).loc(0.5).stimulate(stim)

    # record all path compartments in order
    # apical trunk starts with SOMA comp 0 so out[0] serves as the soma voltage for spike detection
    rec_map: list[tuple[int, int]] = []   # (path_idx, comp_idx_in_path)
    for p_idx, (_, comp_list) in enumerate(paths):
        for c_idx, (br, c, _) in enumerate(comp_list):
            net.cell(1).branch(br).comp(c).record("v")
            rec_map.append((p_idx, c_idx))

    out = np.array(jx.integrate(net, delta_t=dt, t_max=t_max))

    # find soma spike
    v_soma = out[0, :-1]
    idx0 = int(T_EXC_ON / dt)
    cross = np.where(
        (v_soma[idx0:-1] < SOMA_SPIKE_THRESH) & (v_soma[idx0 + 1:] >= SOMA_SPIKE_THRESH)
    )[0]
    if len(cross) == 0:
        raise RuntimeError("Soma never spiked — higher G_SOMA_AMPA?")
    spike_idx = idx0 + cross[0]
    soma_spike_ms = float(cross[0] * dt)

    # time window
    i_start = max(0, spike_idx - int(PRE_SPIKE_MS / dt))
    i_end = min(out.shape[1] - 1, spike_idx + int(POST_SPIKE_MS / dt))
    t_rel = (np.arange(i_start, i_end) - spike_idx) * dt

    # fill voltage matrices
    path_matrices: list[np.ndarray | None] = [None] * len(paths)
    for rec_i, (p_idx, c_idx) in enumerate(rec_map):
        v = out[rec_i, i_start:i_end]   # direct index; no offset needed
        n_comps = len(paths[p_idx][1])
        if path_matrices[p_idx] is None:
            path_matrices[p_idx] = np.zeros((n_comps, len(t_rel)))
        path_matrices[p_idx][c_idx] = v

    return t_rel, path_matrices, soma_spike_ms


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    # convert compartment centre positions to cell boundary positions for pcolormesh
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    diffs  = np.diff(centers)
    edges  = np.empty(len(centers) + 1)
    edges[1:-1] = centers[:-1] + diffs / 2
    edges[0]    = centers[0]  - diffs[0]  / 2
    edges[-1]   = centers[-1] + diffs[-1] / 2
    return edges


def plot_heatmap(
    t_rel: np.ndarray,
    path_matrices: list[np.ndarray],
    paths: list[tuple[str, list[tuple[int, int, float]]]],
    *,
    out_path: str = "",
) -> plt.Figure:
    n_panels = len(paths)

    # width proportional to distance span of each path
    dist_spans = [
        paths[i][1][-1][2] - paths[i][1][0][2] + 50
        for i in range(n_panels)
    ]
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(sum(s / 60 for s in dist_spans) + 3, 6),
        gridspec_kw={"width_ratios": dist_spans},
    )

    vmin, vmax = -80.0, 40.0
    cmap = "RdYlBu_r"   # blue = hyperpolarised, red = depolarised

    t_edges = _cell_edges(t_rel)
    im = None

    for ax, (label, comp_list), mat in zip(axes, paths, path_matrices):
        distances  = np.array([d for _, _, d in comp_list])
        dist_edges = _cell_edges(distances)

        # mat shape: (n_comps, n_times) → pcolormesh expects rows=y, cols=x
        im = ax.pcolormesh(
            dist_edges, t_edges,
            mat.T,           # (n_times, n_comps)
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            shading="flat",
        )

        ax.axhline(0, color="k", lw=1.0, ls="--", alpha=0.7, label="soma spike")
        ax.set_xlabel("Distance from soma centre (μm)", fontsize=9)
        ax.set_title(label, fontsize=9, pad=6)

    axes[0].set_ylabel("Time from soma spike (ms)", fontsize=9)

    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("Membrane voltage (mV)", fontsize=9)
    cbar.ax.axhline(y=-55, color="k", lw=0.6, ls="--", alpha=0.5)   # approx threshold marker

    fig.suptitle(
        f"bAP propagation with G_AMPA = {G_SOMA_AMPA * 1000:.0f} nS \n"
        f"dashed line = soma spike time (at {soma_spike_ms:.2f} ms)",
        fontsize=10, y=1.02,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig



if __name__ == "__main__":
    out_path = os.path.join("results", "calcium_model", "bap_propagation.png")
    paths = build_paths()
    hh_lat = measure_pre_latency(build_pre_cell, T_EXC_ON, dt, t_max)
    print(f"HH latency: {hh_lat:.2f} ms\n")
    t_rel, path_matrices, soma_spike_ms = run_simulation(hh_lat, paths)
    print(f"Soma spike: {soma_spike_ms:.2f} ms after excitation onset\n")
    plot_heatmap(t_rel, path_matrices, paths, out_path=out_path)
