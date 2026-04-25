import os
import sys
import warnings
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["JAX_PLATFORMS"] = "cpu"
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

import numpy as np
import jaxley as jx
from jaxley.connect import connect

from examples.Wilmes_2016.biophys.jaxley import (
    build_wilmes_cell,
    build_pre_cell,
    SOMA, AIS, APROX, APICAL, OBLIQUE,
    BASAL_MAIN, BASAL_BR, SECOND_BASAL, TUFT, SECOND_BR,
    ALL_BRANCHES, NON_SOMA,
    AMPA,
)
from examples.Wilmes_2016.biophys.jaxley.cells import branch_specs

DELTA_T  = 0.025   
T_MAX    = 1200.0
T_WARMUP = 200.0 

N_BG_GRID    = [50, 100, 150, 200]  # number of background AMPA synapses
RATE_BG_GRID = [0.25, 0.5, 1.0, 2.0]       # Poisson rate per synapse (Hz)
N_SEEDS      = 10


G_MEDIAN_NS = 0.3   # nS — target median AMPA conductance
SIGMA_LOG   = 0.5   # log-space standard deviation (moderate skew)
G_MIN_NS    = 0.05  # nS — hard clip lower
G_MAX_NS    = 1.0   # nS — hard clip upper

# ── Pre-cell spike parameters ─────────────────────────────────────────────────

I_AMP_PRE = 0.5  # nA — current to reliably trigger one HH spike
I_DUR_PRE = 0.5  # ms — pulse width

# ── Acceptance / metric parameters ───────────────────────────────────────────

DVDT_THRESH        = 20.0  # mV/ms — dV/dt criterion for spike onset detection
WINDOW_MS          = 50.0  # ms    — non-overlapping window size for spike-free analysis
NEAR_THRESH_MARGIN = 5.0   # mV    — max(V) within this of threshold → near-threshold

# ── Output ────────────────────────────────────────────────────────────────────

RESULTS_DIR      = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
    "results", "context_model_1",
)
EXCLUDE_BRANCHES = [SOMA, AIS]

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Location sampling
# ─────────────────────────────────────────────────────────────────────────────

def get_dendritic_compartments(exclude_branches=None):
    if exclude_branches is None:
        exclude_branches = EXCLUDE_BRANCHES
    comps = []
    for b_idx in sorted(branch_specs.keys()):
        if b_idx in exclude_branches:
            continue
        _, _, ncomp = branch_specs[b_idx]
        for c_idx in range(ncomp):
            comps.append((b_idx, c_idx))
    return comps


def sample_dendritic_locations(n_bg, rng, exclude_branches=None):
    pool = get_dendritic_compartments(exclude_branches)
    if n_bg > len(pool):
        raise ValueError(
            f"Requested n_bg={n_bg}, but only {len(pool)} eligible compartments."
        )
    indices = rng.choice(len(pool), size=n_bg, replace=False)
    return [pool[i] for i in indices]


def draw_log_normal_conductances(n_bg, g_median_ns, sigma_log, g_min_ns, g_max_ns, rng):
    mu = np.log(g_median_ns)
    g = rng.lognormal(mean=mu, sigma=sigma_log, size=n_bg)
    return np.clip(g, g_min_ns, g_max_ns)

def sample_poisson_event_steps(rate_hz, t_max_ms, delta_t, rng):
    if rate_hz <= 0.0:
        return np.array([], dtype=np.int32)

    mean_isi_ms = 1000.0 / rate_hz
    t_ms = 0.0
    steps = []

    while True:
        t_ms += rng.exponential(scale=mean_isi_ms)
        if t_ms >= t_max_ms:
            break
        step = int(round(t_ms / delta_t))
        if step < int(round(t_max_ms / delta_t)):
            steps.append(step)

    return np.asarray(steps, dtype=np.int32)


def events_to_current(event_steps, t_max_ms, delta_t, i_amp, i_dur_ms):
    n_steps = int(round(t_max_ms / delta_t))
    current = np.zeros(n_steps, dtype=np.float32)
    n_dur_steps = max(1, int(round(i_dur_ms / delta_t)))

    for step in event_steps:
        end_step = min(step + n_dur_steps, n_steps)
        current[step:end_step] += i_amp

    return current


def make_poisson_current(rate_hz, t_max_ms, delta_t, i_amp, i_dur_ms, rng):
    event_steps = sample_poisson_event_steps(rate_hz, t_max_ms, delta_t, rng)
    return events_to_current(event_steps, t_max_ms, delta_t, i_amp, i_dur_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Network assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_background_network(cell, n_bg, syn_locs, g_ampa_ns_vec, currents, delta_t, t_max):
    """
    Build a jx.Network with:
      - cell(0): Wilmes postsynaptic cell (18-branch, n_seg=151)
      - cell(1 .. n_bg): single-compartment HH pre-cells

    n_bg AMPA edges: pre-cell i+1 → post-cell dendritic location syn_locs[i].
    Each pre-cell is stimulated with its Poisson current train.

    Recording sites on post-cell (indices in returned v array):
      0: soma    (branch SOMA, loc 0.5)
      1: apical  (branch APICAL, loc 0.9)
      2: oblique (branch OBLIQUE, loc 0.5)
      3: basal   (branch BASAL_MAIN, loc 0.5)
    """
    pre_cells = [build_pre_cell() for _ in range(n_bg)]
    net = jx.Network([cell] + pre_cells)
    net.pumped_ions = list({p.ion_name for p in net.pumps})

    for i, (b_idx, c_idx) in enumerate(syn_locs):
        connect(
            net.cell(i + 1).branch(0).loc(0.5),       # pre
            net.cell(0).branch(b_idx).comp(c_idx),     # post
            AMPA(),
        )
        net.select(edges=i).set("AMPA_gAMPA", float(g_ampa_ns_vec[i]) * 1e-3)  # nS → µS
        net.cell(i + 1).branch(0).loc(0.5).stimulate(currents[i])

    # recording sites on post-cell
    net.cell(0).branch(SOMA).loc(0.5).record("v")
    net.cell(0).branch(APICAL).loc(0.9).record("v")
    net.cell(0).branch(OBLIQUE).loc(0.5).record("v")
    net.cell(0).branch(BASAL_MAIN).loc(0.5).record("v")

    return net


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Metrics
# ─────────────────────────────────────────────────────────────────────────────

def detect_spike_threshold(v_soma, delta_t, dvdt_thresh=DVDT_THRESH):
    """
    Find spike onset times (ms) and estimate threshold voltage (mV).

    Spike onset: first step where dV/dt >= dvdt_thresh mV/ms, after the
    previous spike has repolarised (dV/dt < 0).

    Returns:
        spike_times_ms : np.array of onset times in ms (empty if none)
        threshold_mv   : median voltage at spike onset (NaN if no spikes)
    """
    dvdt         = np.diff(np.asarray(v_soma, dtype=np.float64)) / delta_t
    onsets       = []
    thresh_volts = []
    in_spike     = False

    for k in range(len(dvdt)):
        if not in_spike and dvdt[k] >= dvdt_thresh:
            onsets.append(k * delta_t)
            thresh_volts.append(float(v_soma[k]))
            in_spike = True
        elif in_spike and dvdt[k] < 0.0:
            in_spike = False

    spike_times_ms = np.array(onsets)
    threshold_mv   = float(np.median(thresh_volts)) if thresh_volts else float("nan")
    return spike_times_ms, threshold_mv


def compute_run_metrics(v_soma, delta_t, t_warmup_ms=T_WARMUP, window_ms=WINDOW_MS,
                        near_thresh_margin=NEAR_THRESH_MARGIN):
    """
    Compute operating-regime metrics from the soma voltage trace.

    Discards the first t_warmup_ms ms, then analyses the remainder.

    Returns dict:
        spike_count, spike_rate_hz, spike_threshold_mv,
        n_windows_total, n_windows_spike_free, frac_windows_spike_free,
        n_near_threshold, frac_near_threshold

    frac_near_threshold: fraction of spike-free 50 ms windows where
    max(V) is within near_thresh_margin mV of the detected threshold.
    """
    warmup_steps = int(round(t_warmup_ms / delta_t))
    v_analysis   = np.asarray(v_soma, dtype=np.float64)[warmup_steps:]
    t_analysis_s = len(v_analysis) * delta_t / 1000.0

    spike_times_ms, threshold_mv = detect_spike_threshold(v_analysis, delta_t)

    spike_count  = len(spike_times_ms)
    spike_rate_hz = spike_count / t_analysis_s if t_analysis_s > 0 else float("nan")

    n_steps_per_window = int(round(window_ms / delta_t))
    n_windows_total    = len(v_analysis) // n_steps_per_window

    n_windows_spike_free = 0
    n_near_threshold     = 0

    for w in range(n_windows_total):
        t_win_start = w * window_ms
        t_win_end   = t_win_start + window_ms
        in_window   = (spike_times_ms >= t_win_start) & (spike_times_ms < t_win_end)
        if not np.any(in_window):
            n_windows_spike_free += 1
            v_win = v_analysis[w * n_steps_per_window: (w + 1) * n_steps_per_window]
            if not np.isnan(threshold_mv):
                if np.max(v_win) >= threshold_mv - near_thresh_margin:
                    n_near_threshold += 1

    frac_spike_free  = (n_windows_spike_free / n_windows_total
                        if n_windows_total > 0 else float("nan"))
    frac_near_thresh = (n_near_threshold / n_windows_spike_free
                        if n_windows_spike_free > 0 else float("nan"))

    return {
        "spike_count":             spike_count,
        "spike_rate_hz":           spike_rate_hz,
        "spike_threshold_mv":      threshold_mv,
        "n_windows_total":         n_windows_total,
        "n_windows_spike_free":    n_windows_spike_free,
        "frac_windows_spike_free": frac_spike_free,
        "n_near_threshold":        n_near_threshold,
        "frac_near_threshold":     frac_near_thresh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Single-condition runner
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(N_bg, rate_bg_hz, seed, delta_t=DELTA_T, t_max=T_MAX):
    """
    Run one (N_bg, rate_bg_hz, seed) background condition.

    Builds cell → samples locations → draws conductances → generates Poisson
    currents → assembles network → integrates → computes metrics.

    Returns a dict with all metadata, conductance draws, synapse locations,
    voltage traces, and per-run metrics.
    """
    print(f"  seed {seed:2d} ...", end=" ", flush=True)

    rng = np.random.default_rng(seed)

    cell      = build_wilmes_cell()
    syn_locs  = sample_dendritic_locations(N_bg, rng)
    g_ampa_ns = draw_log_normal_conductances(
        N_bg, G_MEDIAN_NS, SIGMA_LOG, G_MIN_NS, G_MAX_NS, rng
    )
    currents = [
        make_poisson_current(rate_bg_hz, t_max, delta_t, I_AMP_PRE, I_DUR_PRE, rng)
        for _ in range(N_bg)
    ]

    net   = build_background_network(cell, N_bg, syn_locs, g_ampa_ns, currents, delta_t, t_max)
    v     = jx.integrate(net, delta_t=delta_t, t_max=t_max)
    v_arr = np.asarray(v)

    v_soma    = v_arr[0]
    v_apical  = v_arr[1]
    v_oblique = v_arr[2]
    v_basal   = v_arr[3]

    metrics = compute_run_metrics(v_soma, delta_t)

    print(
        f"spikes={metrics['spike_count']:3d}  "
        f"rate={metrics['spike_rate_hz']:5.2f} Hz  "
        f"sf={metrics['frac_windows_spike_free']:.2f}  "
        f"nt={metrics['frac_near_threshold']:.2f}"
    )

    return {
        "seed":          seed,
        "N_bg":          N_bg,
        "rate_bg_hz":    rate_bg_hz,
        "g_ampa_ns":     g_ampa_ns,
        "syn_locs":      syn_locs,
        "v_soma":        v_soma,
        "v_apical":      v_apical,
        "v_oblique":     v_oblique,
        "v_basal":       v_basal,
        **metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Sweep and save
# ─────────────────────────────────────────────────────────────────────────────

def _check_acceptance(seed_results):
    """
    Return True if this condition meets all acceptance criteria across seeds:
      1. mean spike rate in [1, 10] Hz
      2. mean fraction of spike-free 50 ms windows >= 0.30
      3. at least one seed has any near-threshold spike-free windows
    """
    rates = [r["spike_rate_hz"] for r in seed_results]
    sf    = [r["frac_windows_spike_free"] for r in seed_results]
    nt    = [r["frac_near_threshold"] for r in seed_results]

    mean_rate = float(np.nanmean(rates))
    mean_sf   = float(np.nanmean(sf))
    any_nt    = any(
        (not np.isnan(v)) and v > 0.0
        for v in nt
    )
    return (1.0 <= mean_rate <= 10.0) and (mean_sf >= 0.30) and any_nt


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_rows       = []
    accepted_condition = None

    for N_bg in N_BG_GRID:
        for rate in RATE_BG_GRID:
            print(f"\n{'─' * 66}")
            print(f"  N_bg={N_bg}   rate={rate} Hz/syn")
            print(f"{'─' * 66}")
            seed_results = []

            for seed in range(N_SEEDS):
                res = run_condition(N_bg, rate, seed)
                seed_results.append(res)

                # save per-run .npz
                fname = f"N{N_bg}_r{rate}_s{seed}.npz"
                np.savez(
                    os.path.join(RESULTS_DIR, fname),
                    seed=np.int32(seed),
                    N_bg=np.int32(N_bg),
                    rate_bg_hz=np.float32(rate),
                    g_ampa_ns=res["g_ampa_ns"].astype(np.float32),
                    syn_locs=np.array(res["syn_locs"], dtype=np.int32),
                    v_soma=res["v_soma"].astype(np.float32),
                    v_apical=res["v_apical"].astype(np.float32),
                    v_oblique=res["v_oblique"].astype(np.float32),
                    v_basal=res["v_basal"].astype(np.float32),
                    spike_count=np.int32(res["spike_count"]),
                    spike_rate_hz=np.float32(res["spike_rate_hz"]),
                    spike_threshold_mv=np.float32(res["spike_threshold_mv"]),
                    frac_windows_spike_free=np.float32(res["frac_windows_spike_free"]),
                    frac_near_threshold=np.float32(res["frac_near_threshold"]),
                )

                summary_rows.append({
                    "N_bg":                    N_bg,
                    "rate_bg_hz":              rate,
                    "seed":                    seed,
                    "spike_count":             res["spike_count"],
                    "spike_rate_hz":           res["spike_rate_hz"],
                    "spike_threshold_mv":      res["spike_threshold_mv"],
                    "frac_windows_spike_free": res["frac_windows_spike_free"],
                    "frac_near_threshold":     res["frac_near_threshold"],
                })

            # condition-level summary
            mean_rate = float(np.nanmean([r["spike_rate_hz"] for r in seed_results]))
            mean_sf   = float(np.nanmean([r["frac_windows_spike_free"] for r in seed_results]))
            nt_vals   = [r["frac_near_threshold"] for r in seed_results
                         if not np.isnan(r["frac_near_threshold"])]
            mean_nt   = float(np.nanmean(nt_vals)) if nt_vals else float("nan")
            accepted  = _check_acceptance(seed_results)

            print(
                f"\n  CONDITION SUMMARY  "
                f"mean_rate={mean_rate:.2f} Hz  "
                f"mean_sf={mean_sf:.2f}  "
                f"mean_nt={mean_nt:.2f}  "
                f"ACCEPTED={accepted}"
            )

            if accepted and accepted_condition is None:
                accepted_condition = (N_bg, rate)
                print(f"\n  *** FIRST ACCEPTED: N_bg={N_bg}, rate={rate} Hz/syn ***")

    # save summary CSV
    df       = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved → {csv_path}")

    # acceptance table
    print("\n" + "=" * 72)
    print("ACCEPTANCE TABLE")
    print("=" * 72)
    print(f"{'N_bg':>6}  {'rate(Hz)':>8}  {'mean_rate':>10}  {'mean_sf':>8}  "
          f"{'mean_nt':>8}  {'accepted':>8}")
    print("-" * 72)

    cond_groups = df.groupby(["N_bg", "rate_bg_hz"])
    for (n, r), grp in cond_groups:
        seed_dicts = grp[
            ["spike_rate_hz", "frac_windows_spike_free", "frac_near_threshold"]
        ].to_dict("records")
        acc     = _check_acceptance(seed_dicts)
        m_rate  = grp["spike_rate_hz"].mean()
        m_sf    = grp["frac_windows_spike_free"].mean()
        nt_vals = grp["frac_near_threshold"].dropna()
        m_nt    = nt_vals.mean() if len(nt_vals) else float("nan")
        print(f"{int(n):>6}  {r:>8.2f}  {m_rate:>10.2f}  {m_sf:>8.2f}  "
              f"{m_nt:>8.2f}  {str(acc):>8}")

    print("=" * 72)
    if accepted_condition:
        print(
            f"\nFREEZE MODEL 1 BACKGROUND:  "
            f"N_bg={accepted_condition[0]},  "
            f"rate={accepted_condition[1]} Hz/syn\n"
        )
    else:
        print(
            "\nNo condition met all acceptance criteria.\n"
            "Inspect summary.csv and expand the grid or adjust thresholds.\n"
        )


if __name__ == "__main__":
    main()
