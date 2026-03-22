"""
Use this: 
    python archetype_example_traces_inh.py --path apical  --dist 300 --ns 10
    python archetype_example_traces_inh.py --path oblique --dist 160 --ns 8
    python archetype_example_traces_inh.py --path basal   --dist 100 --ns 5

  Panel 1 – Soma voltage V(t)
  Panel 2 – Target site voltage V(t)
  Panel 3 – Synaptic gating variable R(t) - Destexhe kinetic model: dR/dt driven by transmitter C(t)
  Panel 4 – Synaptic current at target site I_syn(t) = g_bar × R(t) × (V_target − E_AMPA)
  Panel 5 – Inhibitory conductance g_inh(t) = G_INH × R_inh(t) at APROX (GABAa)

  Blue dashed line  : excitatory synapse onset T_EXC_ON
  Magenta dash-dot  : inhibitory synapse onset T_INH_ON (= soma spike + INH_DELAY_MS)
  Orange solid line : soma spike time (if AP+)
  Green shade       : bAP search window [T_spike, T_spike + BAP_WINDOW_MS]
  Grey dotted line  : bAP voltage threshold (panels 1–2)
  Regime label      : AP− | AP+/bAP+ | AP+/bAP−  (panel 1)
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JAX_PLATFORMS"] = "cpu"
warnings.filterwarnings("ignore")

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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

INH_BRANCH = APROX
INH_LOC = 0.9
G_INH_NS = 60.0
G_INH_US = G_INH_NS / 1000.0
E_GABA = -73.0         # mV
INH_DELAY_MS = -0.5    # ms relative to soma spike

INH_DIST_UM = dist_start[INH_BRANCH] + INH_LOC * branch_specs[INH_BRANCH][0]

SPIKE_CROSS = 0.0
SPIKE_PEAK = 20.0
REFRACTORY_MS = 2.0
T_INIT = T_EXC_ON - 5.0

BAP_DISTAL_THRESH = -20.0
BAP_WINDOW_MS = 10.0

PRE_WINDOW_MS  = 5.0
POST_WINDOW_MS = 30.0

# AMPA kinetics
_ampa_defaults = AMPA().synapse_params
AMPA_Cmax  = _ampa_defaults["AMPA_Cmax"]
AMPA_Cdur  = _ampa_defaults["AMPA_Cdur"]
AMPA_alpha = _ampa_defaults["AMPA_alpha"]
AMPA_beta  = _ampa_defaults["AMPA_beta"]
AMPA_E_rev = _ampa_defaults["AMPA_eAMPA"]

# GABAa kinetics
_gaba_defaults = GABAa().synapse_params
GABA_Cmax  = _gaba_defaults["GABAa_Cmax"]
GABA_Cdur  = _gaba_defaults["GABAa_Cdur"]
GABA_alpha = _gaba_defaults["GABAa_alpha"]
GABA_beta  = _gaba_defaults["GABAa_beta"]


def _comps_on_branch(br: int) -> list[tuple[int, int]]:
    return [(br, c) for c in range(branch_specs[br][2])]

PATHS = {
    "apical": (
        "Apical trunk  (APROX → APICAL → TUFT)",
        _comps_on_branch(APROX)
        + _comps_on_branch(APICAL)
        + _comps_on_branch(TUFT[0]),
    ),
    "oblique": (
        "Oblique branch",
        _comps_on_branch(OBLIQUE),
    ),
    "basal": (
        "Basal path  (BASAL_MAIN → BASAL_BR → 2nd_BASAL)",
        _comps_on_branch(BASAL_MAIN)
        + _comps_on_branch(BASAL_BR[0])
        + _comps_on_branch(SECOND_BASAL[0]),
    ),
}


def _comp_dist_um(branch_id: int, comp_id: int) -> float:
    total, _, ncomp = branch_specs[branch_id]
    return dist_start[branch_id] + (comp_id + 0.5) * (total / ncomp)


def find_nearest_comp(path_key: str, target_dist_um: float) -> tuple[tuple[int, int], float]:
    _, comps = PATHS[path_key]
    dists = np.array([_comp_dist_um(b, c) for b, c in comps])
    idx   = int(np.argmin(np.abs(dists - target_dist_um)))
    return comps[idx], float(dists[idx])


def compute_R_analytical(
    t: np.ndarray,
    T_syn: float,
    Cmax: float,
    Cdur: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Destexhe kinetic model (two-phase):
      Phase 1 [T_syn, T_syn + Cdur]:  C = Cmax
        R(t) = R_inf + (R0 - R_inf) * exp(-(t - T_syn) / R_tau)
      Phase 2 [T_syn + Cdur, ∞):  C = 0
        R(t) = R1 * exp(-beta * (t - T_syn - Cdur))
    """
    R_inf = Cmax * alpha / (Cmax * alpha + beta)
    R_tau = 1.0 / (alpha * Cmax + beta)
    R_0   = 0.0

    R = np.zeros_like(t)

    mask1 = (t >= T_syn) & (t < T_syn + Cdur)
    dt1   = t[mask1] - T_syn
    R[mask1] = R_inf + (R_0 - R_inf) * np.exp(-dt1 / R_tau)

    R1 = R_inf + (R_0 - R_inf) * np.exp(-Cdur / R_tau)

    mask2 = t >= T_syn + Cdur
    dt2   = t[mask2] - (T_syn + Cdur)
    R[mask2] = R1 * np.exp(-beta * dt2)

    return R


def _build_network_no_inh(
    branch_id: int, loc: float,
    g_target_us: float, t_pre: float,
) -> jx.Network:
    pre_soma   = build_pre_cell()
    pre_target = build_pre_cell()
    post       = build_wilmes_cell_no_ca()

    net = jx.Network([pre_soma, pre_target, post])

    connect(net.cell(0).branch(0).loc(0.5),
            net.cell(2).branch(SOMA).loc(0.5), AMPA())
    net.select(edges=0).set("AMPA_gAMPA", G_SOMA_US)

    connect(net.cell(1).branch(0).loc(0.5),
            net.cell(2).branch(branch_id).loc(loc), AMPA())
    net.select(edges=1).set("AMPA_gAMPA", g_target_us)

    stim = jx.step_current(
        i_delay=t_pre, i_dur=PRE_STIM_DUR, i_amp=PRE_STIM_AMP,
        delta_t=dt, t_max=t_max,
    )
    net.cell(0).branch(0).loc(0.5).stimulate(stim)
    net.cell(1).branch(0).loc(0.5).stimulate(stim)

    net.cell(2).branch(SOMA).loc(0.5).record("v")
    net.cell(2).branch(branch_id).loc(loc).record("v")
    return net


def _build_network_with_inh(
    branch_id: int, loc: float,
    g_target_us: float, t_pre: float,
) -> jx.Network:
    pre_soma   = build_pre_cell()
    pre_target = build_pre_cell()
    pre_inh    = build_pre_cell()
    post       = build_wilmes_cell_no_ca()

    net = jx.Network([pre_soma, pre_target, pre_inh, post])

    connect(net.cell(0).branch(0).loc(0.5),
            net.cell(3).branch(SOMA).loc(0.5), AMPA())
    net.select(edges=0).set("AMPA_gAMPA", G_SOMA_US)

    connect(net.cell(1).branch(0).loc(0.5),
            net.cell(3).branch(branch_id).loc(loc), AMPA())
    net.select(edges=1).set("AMPA_gAMPA", g_target_us)

    connect(net.cell(2).branch(0).loc(0.5),
            net.cell(3).branch(INH_BRANCH).loc(INH_LOC), GABAa())
    net.select(edges=2).set("GABAa_gGABAa", G_INH_US)
    net.select(edges=2).set("GABAa_eGABAa", E_GABA)

    stim_exc = jx.step_current(
        i_delay=t_pre, i_dur=PRE_STIM_DUR, i_amp=PRE_STIM_AMP,
        delta_t=dt, t_max=t_max,
    )
    net.cell(0).branch(0).loc(0.5).stimulate(stim_exc)
    net.cell(1).branch(0).loc(0.5).stimulate(stim_exc)
    # pre_inh stimulus will be injected via data_stimulate (spike-time dependent)

    net.cell(3).branch(SOMA).loc(0.5).record("v")
    net.cell(3).branch(branch_id).loc(loc).record("v")
    return net


def run_trace(
    branch_id: int,
    comp_id: int,
    g_target_ns: float,
    hh_latency: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """
    Two-pass simulation matching archetypes_inh.py's code path exactly:

    Pass 1 — no inhibition, 1-element vmap + data_set (same JIT path as the
             heatmap sweep) to detect soma spike time
    Pass 2 — with inhibition triggered at t_spike + INH_DELAY_MS, also via
             vmap so the bAP detection is numerically identical to the heatmap

    Returns (t, v_soma, v_target, t_inh_onset)
    Voltages come from pass 2 (inhibited run)
    """
    g_target_us = g_target_ns / 1000.0
    loc         = (comp_id + 0.5) / branch_specs[branch_id][2]
    t_pre       = T_EXC_ON - hh_latency
    t_arr       = np.arange(0, t_max, dt)
    g_batch     = jnp.asarray([g_target_us])
    T_DUMMY_INH = t_max + 1000.0

    # pass 1: no inhibition — spike detection via vmap 
    net1 = _build_network_no_inh(branch_id, loc, 0.0, t_pre)

    def sim_pass1(g_uS):
        ps = net1.select(edges=1).data_set("AMPA_gAMPA", g_uS, param_state=None)
        return jx.integrate(net1, param_state=ps, delta_t=dt, t_max=t_max)

    out1 = jax.jit(jax.vmap(sim_pass1))(g_batch)
    out1.block_until_ready()
    v_soma1    = np.asarray(out1[0, 0, :-1])
    t_spike    = detect_first_spike_time(v_soma1, t_arr, T_INIT)
    soma_fired = t_spike is not None

    # pass 2: with inhibition, triggered at spike time 
    if soma_fired:
        t_inh_onset = t_spike + INH_DELAY_MS
        t_inh_pre   = t_inh_onset - hh_latency
    else:
        t_inh_onset = None
        t_inh_pre   = T_DUMMY_INH

    net2 = _build_network_with_inh(branch_id, loc, 0.0, t_pre)

    stim_inh_arr   = jnp.array(jx.step_current(
        i_delay=t_inh_pre, i_dur=PRE_STIM_DUR, i_amp=PRE_STIM_AMP,
        delta_t=dt, t_max=t_max,
    ))
    stim_inh_batch = stim_inh_arr[None, :]  # shape (1, T)

    def sim_pass2(g_uS, stim_inh):
        ps        = net2.select(edges=1).data_set("AMPA_gAMPA", g_uS, param_state=None)
        data_stim = net2.cell(2).branch(0).loc(0.5).data_stimulate(stim_inh)
        return jx.integrate(net2, param_state=ps, data_stimuli=data_stim,
                             delta_t=dt, t_max=t_max)

    out2 = jax.jit(jax.vmap(sim_pass2))(g_batch, stim_inh_batch)
    out2.block_until_ready()

    return (
        t_arr,
        np.asarray(out2[0, 0, :-1]),
        np.asarray(out2[0, 1, :-1]),
        t_inh_onset,
    )


def _annotate_panel(ax, t_spike, t_hi, t_inh_onset=None, show_bap_thresh=True):
    ax.axvline(T_EXC_ON, color="#3498db", lw=1.2, ls="--", zorder=3)
    if t_inh_onset is not None:
        ax.axvline(t_inh_onset, color="magenta", lw=1.2, ls="-.", zorder=3)
    if t_spike is not None:
        ax.axvline(t_spike, color="#e67e22", lw=1.4, ls="-", zorder=3)
        t_bap_end = min(t_spike + BAP_WINDOW_MS, t_hi)
        ax.axvspan(t_spike, t_bap_end, color="#2ecc71", alpha=0.15, zorder=2)
    if show_bap_thresh:
        ax.axhline(BAP_DISTAL_THRESH, color="gray", lw=0.7, ls=":", zorder=1)


def plot_traces(
    t: np.ndarray,
    v_soma: np.ndarray,
    v_target: np.ndarray,
    actual_dist_um: float,
    g_target_ns: float,
    path_key: str,
    t_inh_onset: float | None,
    *,
    out_path: str = "",
) -> plt.Figure:

    # regime detection (using v_soma which comes from the inhibition run)
    t_spike    = detect_first_spike_time(v_soma, t, T_INIT)
    soma_fired = t_spike is not None

    bap_detected = False
    if soma_fired:
        idx0 = int(np.searchsorted(t, t_spike))
        idx1 = min(len(t), idx0 + int(BAP_WINDOW_MS / dt))
        bap_detected = float(np.max(v_target[idx0:idx1])) > BAP_DISTAL_THRESH

    if not soma_fired:
        regime, regime_color = "AP−(soma did not fire)", "#e74c3c"
    elif bap_detected:
        regime, regime_color = "AP+, bAP+ (bAP reached target)", "#2ecc71"
    else:
        regime, regime_color = "AP+, bAP− (bAP did not reach/attenuated)", "#f39c12"

    t_lo = T_EXC_ON - PRE_WINDOW_MS
    t_hi = T_EXC_ON + POST_WINDOW_MS
    mask = (t >= t_lo) & (t <= t_hi)
    t_w  = t[mask]

    # AMPA gating + current
    R_ampa_t = compute_R_analytical(t, T_EXC_ON, AMPA_Cmax, AMPA_Cdur, AMPA_alpha, AMPA_beta)
    R_ampa_w = R_ampa_t[mask]
    g_target_us   = g_target_ns / 1000.0
    I_target_nA   = g_target_us * R_ampa_w * (v_target[mask] - AMPA_E_rev)

    # GABAa gating
    if t_inh_onset is not None:
        R_gaba_t = compute_R_analytical(t, t_inh_onset, GABA_Cmax, GABA_Cdur, GABA_alpha, GABA_beta)
    else:
        R_gaba_t = np.zeros_like(t)
    R_gaba_w    = R_gaba_t[mask]
    g_inh_nS_w  = G_INH_NS * R_gaba_w

    path_label, _ = PATHS[path_key]

    fig, axes = plt.subplots(
        5, 1, figsize=(10, 13.5), sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.8, 1.2, 1.2, 1.2]},
    )
    fig.subplots_adjust(hspace=0.22, top=0.80)

    ax = axes[0]
    ax.plot(t_w, v_soma[mask], color="#2c3e50", lw=1.4)
    _annotate_panel(ax, t_spike, t_hi, t_inh_onset, show_bap_thresh=False)
    ax.set_ylabel("mV", fontsize=9)
    ax.set_title("Soma voltage", fontsize=9, pad=3)
    ax.set_ylim(-85, 55)
    ax.text(0.01, 0.96, regime, transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=regime_color, ec="none", alpha=0.9))

    ax = axes[1]
    ax.plot(t_w, v_target[mask], color="#8e44ad", lw=1.4,
            label=f"Target ({actual_dist_um:.0f} µm, {path_key})")
    _annotate_panel(ax, t_spike, t_hi, t_inh_onset, show_bap_thresh=True)
    ax.set_ylabel("mV", fontsize=9)
    ax.set_title(
        f"Target site voltage ({path_label}, {actual_dist_um:.0f} µm)  "
        f"grey dotted = bAP threshold ({BAP_DISTAL_THRESH} mV)",
        fontsize=9, pad=3,
    )
    ax.set_ylim(-85, 55)

    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(t_w, R_ampa_w, color="#e67e22", lw=1.6)
    ax2.plot(t_w, g_target_ns * R_ampa_w, color="#e67e22", lw=0, alpha=0)
    _annotate_panel(ax, t_spike, t_hi, t_inh_onset, show_bap_thresh=False)
    ax.set_ylabel("R  (fraction open)", fontsize=8.5)
    ax.set_ylim(-0.02, 1.05)
    ax2.set_ylabel("g_syn = g_bar × R (nS)", fontsize=8.5)
    ax2.set_ylim(-0.02 * g_target_ns, 1.05 * g_target_ns)
    ax.set_title(
        "AMPA gating variable R(t)\n"
        f"α={AMPA_alpha}, β={AMPA_beta}, Cmax={AMPA_Cmax}, Cdur={AMPA_Cdur} ms",
        fontsize=8.5, pad=3,
    )

    ax = axes[3]
    ax.plot(t_w, I_target_nA, color="#27ae60", lw=1.4,
            label=f"I_syn = {g_target_ns:.1f} nS × R(t) × (V_target − {AMPA_E_rev} mV)")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    _annotate_panel(ax, t_spike, t_hi, t_inh_onset, show_bap_thresh=False)
    ax.set_ylabel("I_syn (nA)", fontsize=9)
    ax.set_title(
        "Synaptic current at target: I = g_bar × R(t) × (V − E_AMPA)",
        fontsize=8.5, pad=3,
    )

    ax = axes[4]
    ax.plot(t_w, g_inh_nS_w, color="magenta", lw=1.6)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    _annotate_panel(ax, t_spike, t_hi, t_inh_onset, show_bap_thresh=False)
    ax.set_ylabel("g_inh (nS)", fontsize=9)
    ax.set_xlabel("Time (ms)", fontsize=9)
    inh_label = (
        f"GABAa inhibitory conductance at APROX {INH_LOC} ({INH_DIST_UM:.0f} µm)\n"
        f"g_bar = {G_INH_NS:.0f} nS, E_GABA = {E_GABA} mV, "
        f"onset = spike {INH_DELAY_MS:+.1f} ms"
    )
    ax.set_title(inh_label, fontsize=8.5, pad=3)

    fig.suptitle(
        f"Traces (WITH inhibition) — {path_key} path, target at {actual_dist_um:.0f} µm, "
        f"{g_target_ns:.1f} nS\n"
        f"soma AMPA = {G_SOMA_NS:.0f} nS (fixed)  |  "
        f"GABAa = {G_INH_NS:.0f} nS at APROX {INH_LOC} ({INH_DIST_UM:.0f} µm), "
        f"{INH_DELAY_MS:+.1f} ms after spike",
        fontsize=10,
        y=0.88,
    )

    t_spike_str = f"{t_spike:.2f} ms" if soma_fired else "no spike"
    t_inh_str   = f"{t_inh_onset:.2f} ms" if t_inh_onset is not None else "N/A"
    legend_handles = [
        mlines.Line2D([0], [0], color="#3498db", lw=1.2, ls="--",
                      label="Excitatory synapse onset (T_EXC_ON)"),
        mlines.Line2D([0], [0], color="magenta", lw=1.2, ls="-.",
                      label=f"Inhibitory synapse onset (T = {t_inh_str})"),
        mlines.Line2D([0], [0], color="#e67e22", lw=1.4, ls="-",
                      label=f"Soma spike (T = {t_spike_str})"),
        mpatches.Patch(fc="#2ecc71", alpha=0.4,
                       label=f"bAP search window ({BAP_WINDOW_MS:.0f} ms)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=7.5,
        framealpha=0.75,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot voltage + synaptic conductance traces WITH inhibition"
    )
    parser.add_argument("--path", required=True, choices=list(PATHS.keys()))
    parser.add_argument("--dist", required=True, type=float,
                        help="Target synapse distance from soma (µm)")
    parser.add_argument("--ns", required=True, type=float,
                        help="Target synapse conductance (nS)")
    args = parser.parse_args()

    (branch_id, comp_id), actual_dist = find_nearest_comp(args.path, args.dist)
    print(f"Target: branch {branch_id}, comp {comp_id} "
          f"(requested {args.dist:.0f} µm → snapped to {actual_dist:.1f} µm)")
    print(f"Inhibition: APROX {INH_LOC} ({INH_DIST_UM:.1f} µm), "
          f"{G_INH_NS:.0f} nS GABAa, {INH_DELAY_MS:+.1f} ms after spike\n")

    hh_latency = measure_pre_latency(build_pre_cell, T_EXC_ON, dt, t_max)
    print(f"HH latency = {hh_latency:.2f} ms\n")

    t, v_soma, v_target, t_inh_onset = run_trace(
        branch_id, comp_id, args.ns, hh_latency
    )
    print(f"t_inh_onset = {t_inh_onset}\n")

    out_path = os.path.join(
        "results", "calcium_free_model", "archetypes_inh", "example_traces",
        f"{args.path}_{actual_dist:.0f}um_{args.ns:.1f}nS.png",
    )
    plot_traces(
        t, v_soma, v_target,
        actual_dist_um=actual_dist,
        g_target_ns=args.ns,
        path_key=args.path,
        t_inh_onset=t_inh_onset,
        out_path=out_path,
    )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
