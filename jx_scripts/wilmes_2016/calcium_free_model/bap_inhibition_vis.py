import argparse
import os
import sys
import time
from typing import Any
import warnings

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JAX_PLATFORMS"] = "cpu"
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jaxley as jx
from jaxley.connect import connect

from examples.Wilmes_2016.biophys.jaxley import (
    build_wilmes_cell_no_ca,
    build_pre_cell,
    AMPA, GABAa,
    SOMA, AIS, APROX, APICAL, OBLIQUE, BASAL_MAIN, TUFT 
)
from utils import measure_pre_latency

dt = 0.1
t_max = 1125.0

T_EXC_ON = 995.0
PRE_STIM_DUR = 0.5
PRE_STIM_AMP = 0.1

G_SOMA_AMPA = 0.007   # uS
E_GABA = -73.0 # mV
SOMA_EXC_BRANCH = SOMA
SOMA_EXC_LOC = 0.5

T_INH_OFFSETS_MS = np.linspace(-1.5, 1.5, 20)   # ms after soma spike
G_GABA_VALUES_US = np.linspace(0.0005, 0.20, 20)  # uS

BAP_SEARCH_WINDOW_MS = 20.0
SOMA_SPIKE_THRESH_MV = 0.0

# (a) no block 
# (b) distal block: inh PAST oblique branch so only blocks distal/tuft path
# (c) apical block: inh BEFORE oblique branch so blocks oblique AND distal
# (d) basal block: inh on basal tree only
INH_CONFIGS = [
    ("no block", None, None),    
    ("distal", APICAL, 0.5),    
    ("apical", APROX, 0.9),    
    ("basal", BASAL_MAIN, 0.5),    
]

# distal: APICAL/tuft — past the oblique branch point
# oblique: the oblique branch itself
# basal: basal dendrite
# axonal: axon initial segment
REC_CONFIGS = [
    ("distal",   APICAL,    0.9),    
    ("oblique", OBLIQUE,   0.5),    
    ("basal",  BASAL_MAIN, 0.5),  
    ("axonal",  AIS,       0.5),    
]


def measure_baseline(hh_latency: float, rec_branch: int, rec_loc: float) -> tuple[float, float, float]:
    pre_exc = build_pre_cell()
    post = build_wilmes_cell_no_ca()
    net = jx.Network([pre_exc, post])
    connect(net.cell(0).branch(0).loc(0.5),
            net.cell(1).branch(SOMA_EXC_BRANCH).loc(SOMA_EXC_LOC), AMPA())
    net.select(edges=0).set("AMPA_gAMPA", G_SOMA_AMPA)
    stim = jx.step_current(
        i_delay=T_EXC_ON - hh_latency, i_dur=PRE_STIM_DUR, i_amp=PRE_STIM_AMP,
        delta_t=dt, t_max=t_max,
    )
    net.cell(0).branch(0).loc(0.5).stimulate(stim)
    net.cell(1).branch(SOMA).loc(0.5).record("v")
    net.cell(1).branch(rec_branch).loc(rec_loc).record("v")

    out = np.array(jx.integrate(net, delta_t=dt, t_max=t_max))
    v_soma, v_rec = out[0, :-1], out[1, :-1]

    idx0 = int(T_EXC_ON / dt)
    soma_cross = np.where((v_soma[idx0:-1] < 0.0) & (v_soma[idx0 + 1:] >= 0.0))[0]
    if len(soma_cross) == 0:
        raise RuntimeError("Soma never fired. Check G_SOMA_AMPA.")
    soma_delay_ms = float(soma_cross[0] * dt)

    idx_spike = idx0 + int(soma_delay_ms / dt)
    window = v_rec[idx_spike: idx_spike + int(BAP_SEARCH_WINDOW_MS / dt)]
    peak_idx = int(np.argmax(window))
    bap_delay_ms = soma_delay_ms + float(peak_idx * dt)
    baseline_peak = float(window[peak_idx])
    return soma_delay_ms, bap_delay_ms, baseline_peak


def build_network(
    t_exc_pre: float,
    inh_branch: int,
    inh_loc: float,
    rec_branch: int,
    rec_loc: float,
) -> jx.Network:
    pre_exc = build_pre_cell()
    pre_inh = build_pre_cell()
    post = build_wilmes_cell_no_ca()

    net = jx.Network([pre_exc, pre_inh, post])

    connect(net.cell(0).branch(0).loc(0.5),
            net.cell(2).branch(SOMA_EXC_BRANCH).loc(SOMA_EXC_LOC), AMPA())
    net.select(edges=0).set("AMPA_gAMPA", G_SOMA_AMPA)

    connect(net.cell(1).branch(0).loc(0.5),
            net.cell(2).branch(inh_branch).loc(inh_loc), GABAa())
    net.select(edges=1).set("GABAa_gGABAa", G_GABA_VALUES_US[0])  # placeholder
    net.select(edges=1).set("GABAa_eGABAa", E_GABA)

    stim_exc = jx.step_current(
        i_delay=t_exc_pre, i_dur=PRE_STIM_DUR, i_amp=PRE_STIM_AMP,
        delta_t=dt, t_max=t_max,
    )
    net.cell(0).branch(0).loc(0.5).stimulate(stim_exc)

    net.cell(2).branch(SOMA).loc(0.5).record("v")
    net.cell(2).branch(rec_branch).loc(rec_loc).record("v")
    return net


def run_sweep_vmap(
    net: jx.Network,
    soma_delay_ms: float,
    hh_latency: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_g = len(G_GABA_VALUES_US)
    n_t = len(T_INH_OFFSETS_MS)

    stim_inh_arrays = jnp.stack([
        jnp.array(jx.step_current(
            i_delay=(T_EXC_ON + soma_delay_ms + t_off) - hh_latency,
            i_dur=PRE_STIM_DUR, i_amp=PRE_STIM_AMP,
            delta_t=dt, t_max=t_max,
        ))
        for t_off in T_INH_OFFSETS_MS
    ])  # (n_t, T)

    g_flat = jnp.repeat(jnp.array(G_GABA_VALUES_US), n_t)
    stim_flat = jnp.tile(stim_inh_arrays, (n_g, 1))

    def simulate_one(g_gaba, stim_inh_current):
        param_state = net.select(edges=1).data_set(
            "GABAa_gGABAa", g_gaba, param_state=None)
        data_stimuli = net.cell(1).branch(0).loc(0.5).data_stimulate(stim_inh_current)
        return jx.integrate(net, param_state=param_state, data_stimuli=data_stimuli,
                            delta_t=dt, t_max=t_max)  # (2, T+1)

    t0 = time.time()
    all_results = np.array(jax.jit(jax.vmap(simulate_one))(g_flat, stim_flat))

    all_results = all_results.reshape(n_g, n_t, 2, -1)
    v_soma = all_results[:, :, 0, :-1]
    v_rec  = all_results[:, :, 1, :-1]

    idx0 = int(T_EXC_ON / dt)
    idx_spike = idx0 + int(soma_delay_ms / dt)
    idx_end = idx_spike + int(BAP_SEARCH_WINDOW_MS / dt)

    bap_peaks = v_rec[:, :, idx_spike:idx_end].max(axis=-1)

    v_s = v_soma[:, :, idx0:-1]
    soma_fired = np.any(
        (v_s[:, :, :-1] < SOMA_SPIKE_THRESH_MV) & (v_s[:, :, 1:] >= SOMA_SPIKE_THRESH_MV),
        axis=-1,
    )
    return bap_peaks, soma_fired


def run_none_sweep(
    baseline_peak: float,
    n_g: int,
    n_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    bap_peaks  = np.full((n_g, n_t), baseline_peak)
    soma_fired = np.ones((n_g, n_t), dtype=bool)
    return bap_peaks, soma_fired


V_REST = -75.0   # mV — used to normalise bAP amplitude


def compute_suppression(bap_peaks: np.ndarray, baseline_peak: float) -> np.ndarray:
    amplitude_baseline = baseline_peak - V_REST   # mV above rest
    return np.clip((baseline_peak - bap_peaks) / amplitude_baseline * 100, 0, 100)


def plot_grid(
    all_results: dict,
    baseline_peaks: dict,
    soma_delays: dict,
    bap_delays: dict,
    *,
    out_path: str = "",
) -> plt.Figure:
    n_rows = len(INH_CONFIGS)
    n_cols = len(REC_CONFIGS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.8 * n_rows),
        sharex=True, sharey=True,
    )

    # single shared colorbar: 0 % (white) to 100 % (blue, fully suppressed)
    cmap = plt.cm.Blues
    vmin_pct, vmax_pct = 0, 100

    # one colorbar on the right edge spanning all rows for the last column
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin_pct, vmax=vmax_pct))
    sm.set_array([])

    for col_idx, (rec_label, _, _) in enumerate(REC_CONFIGS):
        baseline_peak = baseline_peaks[rec_label]
        prop_window = bap_delays[rec_label] - soma_delays[rec_label]

        for row_idx, (inh_label, _, _) in enumerate(INH_CONFIGS):
            ax = axes[row_idx, col_idx]
            bap_peaks, soma_fired = all_results[(inh_label, rec_label)]

            suppression = compute_suppression(bap_peaks, baseline_peak)

            im = ax.imshow(
                suppression,
                origin="lower", aspect="auto",
                extent=[T_INH_OFFSETS_MS[0], T_INH_OFFSETS_MS[-1],
                        G_GABA_VALUES_US[0] * 1000, G_GABA_VALUES_US[-1] * 1000],
                vmin=vmin_pct, vmax=vmax_pct,
                cmap=cmap,
            )

            # grey out soma-suppressed region (separate from bAP suppression)
            if not soma_fired.all():
                ax.contourf(T_INH_OFFSETS_MS, G_GABA_VALUES_US * 1000,
                            (~soma_fired).astype(float),
                            levels=[0.5, 1.5], colors=["0.75"], alpha=0.7, zorder=2)
                ax.contour(T_INH_OFFSETS_MS, G_GABA_VALUES_US * 1000,
                           soma_fired.astype(float),
                           levels=[0.5], colors=["k"],
                           linewidths=1.0, linestyles="--", zorder=3)

            # propagation window markers
            ax.axvline(0, color="0.4", lw=0.9, ls=":", alpha=0.8, zorder=4)
            if prop_window > 0:
                ax.axvline(prop_window, color="0.4", lw=0.9, ls="--", alpha=0.8, zorder=4)
                ax.axvspan(0, prop_window, color="0.4", alpha=0.05, zorder=1)

            # baseline peak annotation in top right corner of each panel
            ax.text(0.97, 0.96, f"base={baseline_peak:.0f} mV",
                    transform=ax.transAxes, fontsize=7,
                    ha="right", va="top", color="0.3")

            # row labels (left column only)
            if col_idx == 0:
                ax.set_ylabel(f"{inh_label}\nG_GABAa (nS)", fontsize=8)
            else:
                ax.set_ylabel("")

            # column labels (top row only)
            if row_idx == 0:
                ax.set_title(f"rec: {rec_label}", fontsize=9)

            # x-axis label (bottom row only)
            if row_idx == n_rows - 1:
                ax.set_xlabel("inh offset re. soma spike (ms)", fontsize=8)

    # single shared colorbar on the right
    cbar = fig.colorbar(sm, ax=axes[:, -1], fraction=0.03, pad=0.02)
    cbar.set_label("bAP suppression (%)\n0 % = baseline  |  100 % = fully blocked", fontsize=9)
    cbar.ax.axhline(y=50, color="k", lw=0.8, ls="--", alpha=0.5)

    fig.suptitle(
        f"bAP suppression grid\n"
        f"Soma G_AMPA = {G_SOMA_AMPA*1000:.0f} nS\n"
        f"rows = inh location, cols = recording site\n"
        f"(grey = soma suppressed)",
        fontsize=11, y=1.01,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    out_path = os.path.join("results", "calcium_free_model", "bap_inh_heatmap.png")

    hh_lat = measure_pre_latency(build_pre_cell, T_EXC_ON, dt, t_max)
    print(f"HH latency: {hh_lat:.2f} ms\n")
    # baseline per recording site
    soma_delays = {}
    bap_delays = {}
    baseline_peaks = {}
    for rec_label, rec_branch, rec_loc in REC_CONFIGS:
        print(f"  rec={rec_label} ...", end=" ")
        sd, bd, bp = measure_baseline(hh_lat, rec_branch, rec_loc)
        soma_delays[rec_label] = sd
        bap_delays[rec_label]= bd
        baseline_peaks[rec_label] = bp
        print(f"soma_delay={sd:.2f} ms, bAP_peak={bp:.1f} mV, prop={bd-sd:.2f} ms")
    print()

    # use OBLIQUE soma_delay as the canonical timing reference
    soma_delay_ref = soma_delays["axonal"]

    all_results = {}
    n_g, n_t = len(G_GABA_VALUES_US), len(T_INH_OFFSETS_MS)

    for inh_label, inh_branch, inh_loc in INH_CONFIGS:
        for rec_label, rec_branch, rec_loc in REC_CONFIGS:
            print(f"inh={inh_label:<8s}, rec={rec_label:<8s}", end=" ")

            if inh_branch is None:
                bap_peaks, soma_fired = run_none_sweep(
                    baseline_peaks[rec_label], n_g, n_t
                )
                print("baseline, no simulation needed")
            else:
                net = build_network(
                    T_EXC_ON - hh_lat,
                    inh_branch, inh_loc,
                    rec_branch, rec_loc,
                )
                bap_peaks, soma_fired = run_sweep_vmap(net, soma_delay_ref, hh_lat)

            all_results[(inh_label, rec_label)] = (bap_peaks, soma_fired)
    plot_grid(
        all_results, baseline_peaks, soma_delays, bap_delays,
        out_path=out_path,
    )
