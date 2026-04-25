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
    SOMA, OBLIQUE, BASAL_MAIN, TUFT,
)
from utils import detect_spikes, measure_pre_latency, classify_response

dt = 0.1 # ms
t_max = 1125.0 # ms

T_SYN_ON = 995.0    # ms 
PRE_STIM_DUR = 0.5   # ms
PRE_STIM_AMP = 0.1   # nA

# sweep range
G_MIN = 0.001   # uS = 1 nS
G_MAX = 0.020   # uS = 20 nS
N_STEPS = 20

# analysis windows
ANALYSIS_WIN_MS = 20.0 # classify first 20 ms after synaptic onset
MIN_SPIKE_PEAK = 20.0 # mV
SPIKE_CROSS = 0.0 # upward crossing threshold
REFRACTORY_MS = 2.0 # merge spikes closer than this

def build_soma_ampa_circuit(g_ampa_uS: float, t_pre_current: float) -> jx.Network:
    pre = build_pre_cell()
    post = build_wilmes_cell()

    net = jx.Network([pre, post])
    net.pumped_ions = list({p.ion_name for p in net.pumps})

    connect(
        net.cell(0).branch(0).loc(0.5),
        net.cell(1).branch(SOMA).loc(0.5),
        AMPA(),
    )
    net.select(edges=0).set("AMPA_gAMPA", g_ampa_uS)

    stim_pre = jx.step_current(
        i_delay=t_pre_current,
        i_dur=PRE_STIM_DUR,
        i_amp=PRE_STIM_AMP,
        delta_t=dt,
        t_max=t_max,
    )
    net.cell(0).branch(0).loc(0.5).stimulate(stim_pre)
    net.cell(1).branch(SOMA).loc(0.5).record("v") # rec 0 soma
    net.cell(1).branch(OBLIQUE).loc(0.5).record("v") # rec 1
    net.cell(1).branch(TUFT[0]).loc(0.5).record("v") # rec 2
    net.cell(1).branch(BASAL_MAIN).loc(0.5).record("v") # rec 3
    net.cell(0).branch(0).loc(0.5).record("v") # rec 4 pre spike

    return net


def summarize_trace(v_soma, v_oblique, v_tuft, v_basal, t, t_syn_on):
    t_end = t_syn_on + ANALYSIS_WIN_MS
    mask = (t >= t_syn_on) & (t <= t_end)

    spike_times, spike_peaks = detect_spikes(v_soma, t, t_syn_on)

    # peak timing of first spike, if any
    first_spike_time = spike_times[0] if len(spike_times) > 0 else None
    first_spike_latency = None if first_spike_time is None else (first_spike_time - t_syn_on)

    # peak values in early window
    peak_soma = float(np.max(v_soma[mask]))
    peak_oblique = float(np.max(v_oblique[mask]))
    peak_tuft = float(np.max(v_tuft[mask]))
    peak_basal = float(np.max(v_basal[mask]))

    metrics = {
        "n_spikes": len(spike_times),
        "spike_times": spike_times,
        "spike_peaks": spike_peaks,
        "first_spike_latency_ms": first_spike_latency,
        "peak_soma_mV": peak_soma,
        "peak_oblique_mV": peak_oblique,
        "peak_tuft_mV": peak_tuft,
        "peak_basal_mV": peak_basal,
    }
    metrics["class"] = classify_response(metrics)
    return metrics

os.makedirs("results/calcium_model/soma_ampa_sweep_ca", exist_ok=True)

# align pre current so postsynaptic AMPA onset is at T_SYN_ON
hh_latency = measure_pre_latency(build_pre_cell, T_SYN_ON, dt, t_max)
t_pre_current = T_SYN_ON - hh_latency
print(f"Measured HH latency: {hh_latency:.2f} ms")
print(f"Using pre current onset at {t_pre_current:.2f} ms so AMPA onset = {T_SYN_ON:.2f} ms")

g_sweep = np.linspace(G_MIN, G_MAX, N_STEPS)
t = np.arange(0, t_max, dt)
results = []

print("\nSweep results")
print("-" * 90)
print(f"{'g (nS)':>8} | {'class':>12} | {'nspk':>4} | {'lat(ms)':>7} | {'Vsoma pk':>8}")

for g in g_sweep:
    net = build_soma_ampa_circuit(g_ampa_uS=float(g), t_pre_current=t_pre_current)
    v = np.array(jx.integrate(net, delta_t=dt, t_max=t_max))

    v_soma = v[0, :-1]
    v_oblique = v[1, :-1]
    v_tuft = v[2, :-1]
    v_basal = v[3, :-1]
    v_pre = v[4, :-1]

    metrics = summarize_trace(v_soma, v_oblique, v_tuft, v_basal, t, T_SYN_ON)

    result = {
        "g_uS": float(g),
        "g_nS": float(g) * 1e3,
        "metrics": metrics,
        "v_soma": v_soma,
        "v_oblique": v_oblique,
        "v_tuft": v_tuft,
        "v_basal": v_basal,
        "v_pre": v_pre,
    }
    results.append(result)

    lat = metrics["first_spike_latency_ms"]
    lat_str = "  --- " if lat is None else f"{lat:7.2f}"
    print(f"{result['g_nS']:8.2f} | {metrics['class']:>12} | {metrics['n_spikes']:4d} | {lat_str} |"
            f" {metrics['peak_soma_mV']:8.2f}")

# all soma traces 
fig, ax = plt.subplots(figsize=(8, 4.5))
for r in results:
    cls = r["metrics"]["class"]
    if cls == "subthreshold":
        color = "steelblue"
        alpha = 0.7
    elif cls == "single_spike":
        color = "tomato"
        alpha = 0.9
    else:
        color = "purple"
        alpha = 0.8

    ax.plot(t, r["v_soma"], color=color, lw=0.9, alpha=alpha)

ax.axvline(T_SYN_ON, color="gray", ls="--", lw=0.8, label="AMPA onset")
ax.set_xlim(T_SYN_ON - 5, T_SYN_ON + 30)
ax.set_ylim(-80, 60)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Somatic voltage (mV)")
ax.set_title("Somatic AMPA sweep: response regimes")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("results/calcium_model/soma_ampa_sweep_ca/soma_traces_regimes.png", dpi=150, bbox_inches="tight")

# peak and latency
g_vals = [r["g_nS"] for r in results]
peak_vals = [r["metrics"]["peak_soma_mV"] for r in results]
lat_vals = [
    np.nan if r["metrics"]["first_spike_latency_ms"] is None else r["metrics"]["first_spike_latency_ms"]
    for r in results
]
colors = []
for r in results:
    cls = r["metrics"]["class"]
    colors.append(
        "steelblue" if cls == "subthreshold"
        else "tomato" if cls == "single_spike"
        else "purple"
    )

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)

ax1.plot(g_vals, peak_vals, color="gray", lw=0.8)
ax1.scatter(g_vals, peak_vals, c=colors, s=45, zorder=3)
ax1.axhline(MIN_SPIKE_PEAK, color="gray", ls="--", lw=0.8, label="spike peak criterion")
ax1.set_ylabel("Peak soma V (mV)")
ax1.set_title("Peak soma voltage vs somatic AMPA")
ax1.legend(fontsize=8)

ax2.scatter(g_vals, lat_vals, c=colors, s=45, zorder=3)
ax2.set_xlabel("AMPA_gAMPA (nS)")
ax2.set_ylabel("1st spike latency (ms)")
ax2.set_title("First-spike latency vs somatic AMPA")
ax2.legend(fontsize=8)

fig2.tight_layout()
fig2.savefig("results/calcium_free_model/soma_ampa_sweep/peak_latency_vs_g.png", dpi=150, bbox_inches="tight")


