"""
Use this: 
    python seep_trace.py --path apical  --dist 300 --ns 10
    python seep_trace.py --path oblique --dist 160 --ns 8
    python seep_trace.py --path basal   --dist 100 --ns 5

  Panel 1 – Soma voltage V(t)
  Panel 2 – Target site voltage V(t)
  Panel 3 – Synaptic gating variable R(t)- Destexhe kinetic model: dR/dt driven by transmitter C(t)
  Panel 4 – Synaptic current at target site I_syn(t) = g_bar × R(t) × (V_target − E_AMPA)

  Blue dashed line : synapse onset T_EXC_ON
  Orange solid line : soma spike time (if AP+)
  Green shade : bAP search window [T_spike, T_spike + BAP_WINDOW_MS]
  Grey dotted line : bAP voltage threshold (panels 1–2)
  Regime label : AP− | AP+/bAP+ | AP+/bAP−  (panel 1)
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

SPIKE_CROSS = 0.0
SPIKE_PEAK = 20.0
REFRACTORY_MS = 2.0
T_INIT = T_EXC_ON - 5.0

BAP_DISTAL_THRESH = -20.0
BAP_WINDOW_MS = 10.0

PRE_WINDOW_MS  = 5.0
POST_WINDOW_MS = 30.0

# ampa kinetics
# I  = g_bar * R(t) * (V - E_AMPA)
# R is the fraction of open channels (0–1) and it is VOLTAGE-INDEPENDENT!!!
# during neurotransmitter C=Cmax for Cdur ms, then C=0 again
_ampa_defaults = AMPA().synapse_params
AMPA_Cmax  = _ampa_defaults["AMPA_Cmax"]
AMPA_Cdur  = _ampa_defaults["AMPA_Cdur"]
AMPA_alpha = _ampa_defaults["AMPA_alpha"]
AMPA_beta  = _ampa_defaults["AMPA_beta"]
AMPA_E_rev = _ampa_defaults["AMPA_eAMPA"]

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


def compute_R_analytical(t: np.ndarray, T_syn: float) -> np.ndarray:
    """
    Kinetics:
      Phase 1  [T_syn, T_syn + Cdur]:  C = Cmax
        R(t) = R_inf + (R0 - R_inf) * exp(-(t - T_syn) / R_tau)
        with R0 = 0 (resting), R_inf = Cmax*alpha/(Cmax*alpha+beta)
             R_tau = 1/(Cmax*alpha+beta)

      Phase 2  [T_syn + Cdur, ∞):  C = 0
        R(t) = R1 * exp(-beta * (t - T_syn - Cdur))
        with R1 = R(T_syn + Cdur) from phase 1
    """
    R_inf = AMPA_Cmax * AMPA_alpha / (AMPA_Cmax * AMPA_alpha + AMPA_beta)
    R_tau = 1.0 / (AMPA_alpha * AMPA_Cmax + AMPA_beta)
    R_0 = 0 # valid bc we spike once/fire once 

    R = np.zeros_like(t)

    # phase 1: transmitter pulse
    mask1 = (t >= T_syn) & (t < T_syn + AMPA_Cdur)
    dt1   = t[mask1] - T_syn
    R[mask1] = R_inf + (R_0 - R_inf) * (np.exp(-dt1 / R_tau))

    # R value at end of phase 1 (phase 1 formula evaluated at dt1 = Cdur)
    R1 = R_inf + (R_0 - R_inf) * np.exp(-AMPA_Cdur / R_tau)

    # phase 2: free decay
    mask2 = t >= T_syn + AMPA_Cdur
    dt2   = t[mask2] - (T_syn + AMPA_Cdur)
    R[mask2] = R1 * np.exp(-AMPA_beta * dt2)

    return R


def run_trace(
    branch_id: int,
    comp_id: int,
    g_target_ns: float,
    hh_latency: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses the same vmap + data_set code path as archetypes_no_inh.py so that
    spike detection at borderline conductances is numerically identical to the
    heatmap (JAX/XLA JIT floating-point behaviour can differ between a plain
    integrate call and a vmapped call)
    """
    g_target_us = g_target_ns / 1000.0
    loc = (comp_id + 0.5) / branch_specs[branch_id][2]
    t_pre = T_EXC_ON - hh_latency

    pre_soma = build_pre_cell()
    pre_target = build_pre_cell()
    post = build_wilmes_cell_no_ca()

    net = jx.Network([pre_soma, pre_target, post])

    connect(net.cell(0).branch(0).loc(0.5),
            net.cell(2).branch(SOMA).loc(0.5), AMPA())
    net.select(edges=0).set("AMPA_gAMPA", G_SOMA_US)

    connect(net.cell(1).branch(0).loc(0.5),
            net.cell(2).branch(branch_id).loc(loc), AMPA())
    net.select(edges=1).set("AMPA_gAMPA", 0.0)  # init to 0; overridden by data_set

    stim = jx.step_current(
        i_delay=t_pre, i_dur=PRE_STIM_DUR, i_amp=PRE_STIM_AMP,
        delta_t=dt, t_max=t_max,
    )
    net.cell(0).branch(0).loc(0.5).stimulate(stim)
    net.cell(1).branch(0).loc(0.5).stimulate(stim)

    net.cell(2).branch(SOMA).loc(0.5).record("v")
    net.cell(2).branch(branch_id).loc(loc).record("v")

    # 1-element vmap — identical JIT compilation path to the heatmap sweep
    g_batch = jnp.asarray([g_target_us])

    def simulate_one(g_uS):
        ps = net.select(edges=1).data_set("AMPA_gAMPA", g_uS, param_state=None)
        return jx.integrate(net, param_state=ps, delta_t=dt, t_max=t_max)

    out = jax.jit(jax.vmap(simulate_one))(g_batch)
    out.block_until_ready()

    t_arr = np.arange(0, t_max, dt)
    return t_arr, np.asarray(out[0, 0, :-1]), np.asarray(out[0, 1, :-1])



def _annotate_panel(ax, t_spike, t_hi, show_bap_thresh=True):
    ax.axvline(T_EXC_ON, color="#3498db", lw=1.2, ls="--", zorder=3)
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
    *,
    out_path: str = "",
) -> plt.Figure:

    # regime detection
    t_spike = detect_first_spike_time(v_soma, t, T_INIT)
    soma_fired = t_spike is not None

    bap_detected = False
    if soma_fired:
        idx0 = int(np.searchsorted(t, t_spike))
        idx1 = min(len(t), idx0 + int(BAP_WINDOW_MS / dt))
        bap_detected = float(np.max(v_target[idx0:idx1])) > BAP_DISTAL_THRESH

    if not soma_fired:
        regime, regime_color = "AP−(soma did not fire)","#e74c3c"
    elif bap_detected:
        regime, regime_color = "AP+, bAP+ (bAP reached target)","#2ecc71"
    else:
        regime, regime_color = "AP+, bAP− (bAP did not reach/attenuated)", "#f39c12"

    t_lo  = T_EXC_ON - PRE_WINDOW_MS
    t_hi  = T_EXC_ON + POST_WINDOW_MS
    mask  = (t >= t_lo) & (t <= t_hi)
    t_w   = t[mask]


    R_t = compute_R_analytical(t, T_EXC_ON)    # full simulation time
    R_w = R_t[mask]

    g_target_us    = g_target_ns / 1000.0
    g_soma_syn_nS  = G_SOMA_NS * R_w  # soma g_syn(t) in nS
    g_target_syn_nS = g_target_ns * R_w   # target g_syn(t) in nS

    I_target_nA = g_target_us * R_t[mask] * (v_target[mask] - AMPA_E_rev)
    # (µS × mV = nA)

    path_label, _ = PATHS[path_key]

    fig, axes = plt.subplots(
        4, 1, figsize=(10, 11), sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.8, 1.2, 1.2]},
    )
    fig.subplots_adjust(hspace=0.2, top=0.76)

    ax = axes[0]
    ax.plot(t_w, v_soma[mask], color="#2c3e50", lw=1.4, label="Soma(V)")
    _annotate_panel(ax, t_spike, t_hi, show_bap_thresh=False)
    ax.set_ylabel("mV", fontsize=9)
    ax.set_title("Soma voltage", fontsize=9, pad=3)
    ax.set_ylim(-85, 55)
    ax.text(0.01, 0.96, regime, transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=regime_color, ec="none", alpha=0.9))

    ax = axes[1]
    ax.plot(t_w, v_target[mask], color="#8e44ad", lw=1.4,
            label=f"Target ({actual_dist_um:.0f} µm, {path_key})")
    _annotate_panel(ax, t_spike, t_hi, show_bap_thresh=True)
    ax.set_ylabel("mV", fontsize=9)
    ax.set_title(
        f"Target site voltage ({path_label}, {actual_dist_um:.0f} µm)  "
        f"grey dotted = bAP threshold ({BAP_DISTAL_THRESH} mV)",
        fontsize=9, pad=3,
    )
    ax.set_ylim(-85, 55)

    ax = axes[2]
    ax2 = ax.twinx()

    ax.plot(t_w, R_w, color="#e67e22", lw=1.6, label="R(t)[dimensionless]")
    ax2.plot(t_w, g_target_syn_nS, color="#e67e22", lw=0, alpha=0)  # invisible; sets scale
    _annotate_panel(ax, t_spike, t_hi, show_bap_thresh=False)

    ax.set_ylabel("R(fraction open channels)", fontsize=8.5)
    ax.set_ylim(-0.02, 1.05)
    ax2.set_ylabel(f"g_syn = g_bar x R (nS)", fontsize=8.5)
    ax2.set_ylim(-0.02 * g_target_ns, 1.05 * g_target_ns)

    # add soma and target g_syn as text on plot
    # R_peak = float(np.max(R_w))
    # ax.annotate(
    #     f"R_peak = {R_peak:.3f}\n"
    #     f"g_soma_peak = {G_SOMA_NS * R_peak:.2f} nS\n"
    #     f"g_target_peak = {g_target_ns * R_peak:.2f} nS",
    #     xy=(T_EXC_ON + AMPA_Cdur * 0.6, R_peak * 0.95),
    #     xytext=(T_EXC_ON + 2.5, R_peak * 0.6),
    #     fontsize=7.5, color="#e67e22",
    #     arrowprops=dict(arrowstyle="->", color="#e67e22", lw=0.8),
    # )
    ax.set_title(
        "Synaptic gating variable R(t)\n"
        f"same curve for both synapses with α={AMPA_alpha}, β={AMPA_beta}, "
        f"Cmax={AMPA_Cmax}, Cdur={AMPA_Cdur} ms)",
        fontsize=8.5, pad=3,
    )

    ax = axes[3]
    ax.plot(t_w, I_target_nA, color="#27ae60", lw=1.4,
            label=f"I_syn = {g_target_ns:.1f} nS × R(t) × (V_target − {AMPA_E_rev} mV)")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    _annotate_panel(ax, t_spike, t_hi, show_bap_thresh=False)

    ax.set_ylabel("I_syn (nA)", fontsize=9)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_title(
        f"Synaptic current at target: I = g_bar × R(t) × (V − E_AMPA)",
        fontsize=8.5, pad=3,
    )
    fig.suptitle(
        f"Traces from {path_key} path with target synapse at {actual_dist_um:.0f} µm and {g_target_ns:.1f} nS conductance\n"
        f"soma conductance of {G_SOMA_NS:.0f} nS (fixed)",
        fontsize=10,
        y=0.83,
    )
    legend_handles = [
        mlines.Line2D([0],[0], color="#3498db", lw=1.2, ls="--", label="Synapse onset (T_EXC_ON)"),
        mlines.Line2D([0],[0], color="#e67e22", lw=1.4, ls="-",
               label=f"Soma spike (T = {t_spike:.2f} ms)" if soma_fired else "No soma spike"),
        mpatches.Patch(fc="#2ecc71", alpha=0.4,
                       label=f"bAP search window ({BAP_WINDOW_MS:.0f} ms)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        framealpha=0.75,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser(
        description="Plot voltage + synaptic conductance traces"
    )
    parser.add_argument("--path", required=True, choices=list(PATHS.keys()))
    parser.add_argument("--dist", required=True, type=float,
                        help="Target synapse distance from soma")
    parser.add_argument("--ns", required=True, type=float,
                        help="Target synapse conductance (nS)")
    args = parser.parse_args()

    (branch_id, comp_id), actual_dist = find_nearest_comp(args.path, args.dist)
    print(f"Target: branch {branch_id}, comp {comp_id}"
          f"(requested {args.dist:.0f} µm snapped to {actual_dist:.1f} µm)")

    R_inf = AMPA_Cmax * AMPA_alpha / (AMPA_Cmax * AMPA_alpha + AMPA_beta)
    R_tau = 1.0 / (AMPA_alpha * AMPA_Cmax + AMPA_beta)
    print(f"AMPA kinetics: R_inf={R_inf:.3f}, R_tau={R_tau:.3f} ms,"
          f"decay tau=1/β={1/AMPA_beta:.2f} ms\n")

    hh_latency = measure_pre_latency(build_pre_cell, T_EXC_ON, dt, t_max)
    print(f"Latency = {hh_latency:.2f} ms\n")

    t, v_soma, v_target = run_trace(branch_id, comp_id, args.ns, hh_latency)

    out_path = os.path.join(
        "results", "calcium_free_model", "archetypes_no_inh", "example_traces",
        f"{args.path}_{actual_dist:.0f}um_{args.ns:.1f}nS.png",
    )
    plot_traces(
        t, v_soma, v_target,
        actual_dist_um=actual_dist,
        g_target_ns=args.ns,
        path_key=args.path,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
