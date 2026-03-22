from __future__ import annotations

import numpy as np
import jaxley as jx


def detect_spikes(
    v: np.ndarray,
    t: np.ndarray,
    t_start: float,
    cross_thresh: float = 0.0,
    peak_thresh: float = 20.0,
    refractory_ms: float = 2.0,
) -> tuple[list[float], list[float]]:
    """Detect all action potentials in a voltage trace.

    Uses upward zero-crossing + local peak amplitude to distinguish true APs
    from subthreshold events.  A refractory window prevents double-counting.

    Parameters
    ----------
    v, t:
        Voltage (mV) and time (ms) arrays of equal length.
    t_start:
        Ignore events before this time (ms).
    cross_thresh:
        Upward-crossing threshold (mV).  Default 0 mV.
    peak_thresh:
        Minimum local peak within 2 ms of crossing to count as a spike (mV).
    refractory_ms:
        Merge detections closer than this apart (ms).

    Returns
    -------
    spike_times : list[float]
        Times (ms) of detected spike peaks.
    spike_peaks : list[float]
        Peak voltages (mV) at those times.
    """
    idx_start = np.searchsorted(t, t_start)
    dt = t[1] - t[0]
    refractory_steps = max(1, int(refractory_ms / dt))
    peak_window_steps = max(1, int(2.0 / dt))

    spike_times: list[float] = []
    spike_peaks: list[float] = []
    last_idx = -(10 ** 9)

    for i in range(idx_start, len(v) - 1):
        if i - last_idx < refractory_steps:
            continue
        if not ((v[i] < cross_thresh) and (v[i + 1] >= cross_thresh)):
            continue

        j1 = i + 1
        j2 = min(len(v), j1 + peak_window_steps)
        local_peak = float(np.max(v[j1:j2]))
        local_peak_idx = j1 + int(np.argmax(v[j1:j2]))

        if local_peak >= peak_thresh:
            spike_times.append(float(t[local_peak_idx]))
            spike_peaks.append(local_peak)
            last_idx = local_peak_idx

    return spike_times, spike_peaks


def detect_first_spike_time(
    v: np.ndarray,
    t: np.ndarray,
    t_start: float,
    cross_thresh: float = 0.0,
    peak_thresh: float = 20.0,
    refractory_ms: float = 2.0,
) -> float | None:
    """Return the time (ms) of the first action potential after t_start, or None.

    Same detection logic as :func:`detect_spikes` but stops at the first event.
    """
    idx_start = np.searchsorted(t, t_start)
    dt = t[1] - t[0]
    ref_steps = max(1, int(refractory_ms / dt))
    peak_steps = max(1, int(2.0 / dt))
    last_idx = -(10 ** 9)

    for i in range(idx_start, len(v) - 1):
        if i - last_idx < ref_steps:
            continue
        if not (v[i] < cross_thresh and v[i + 1] >= cross_thresh):
            continue

        j1 = i + 1
        j2 = min(len(v), j1 + peak_steps)
        peak = float(np.max(v[j1:j2]))
        peak_idx = j1 + int(np.argmax(v[j1:j2]))

        if peak >= peak_thresh:
            return float(t[peak_idx])
        last_idx = peak_idx

    return None


# ── presynaptic HH latency ─────────────────────────────────────────────────────

def measure_pre_latency(
    build_pre_cell_fn,
    t_syn_on: float,
    dt: float,
    t_max: float,
    pre_stim_dur: float = 0.5,
    pre_stim_amp: float = 0.1,
) -> float:
    """Measure latency from current onset to first upward crossing of 0 mV.

    Runs a single isolated presynaptic HH cell and finds when it fires.  The
    result is used to offset the pre-cell stimulation time so that the effective
    synaptic onset in a full network aligns with the desired clock.

    Parameters
    ----------
    build_pre_cell_fn:
        Callable (no arguments) that returns a fresh jaxley presynaptic cell.
    t_syn_on:
        Time (ms) at which the step-current stimulus starts.
    dt, t_max:
        Simulation timestep and duration (ms).
    pre_stim_dur:
        Stimulus duration (ms).  Default 0.5 ms.
    pre_stim_amp:
        Stimulus amplitude (nA).  Default 0.1 nA.

    Returns
    -------
    float
        Latency in ms from current onset to first upward crossing of 0 mV.

    Raises
    ------
    RuntimeError
        If the pre-cell never fires within the simulation window.
    """
    pre = build_pre_cell_fn()
    stim = jx.step_current(
        i_delay=t_syn_on,
        i_dur=pre_stim_dur,
        i_amp=pre_stim_amp,
        delta_t=dt,
        t_max=t_max,
    )
    pre.branch(0).loc(0.5).stimulate(stim)
    pre.branch(0).loc(0.5).record("v")

    v = np.array(jx.integrate(pre, delta_t=dt, t_max=t_max))[0, :-1]
    idx0 = int(t_syn_on / dt)
    crossings = np.where((v[idx0:-1] < 0.0) & (v[idx0 + 1:] >= 0.0))[0]
    if len(crossings) == 0:
        raise RuntimeError(
            "Pre-cell never crossed 0 mV - Increase pre_stim_amp or pre_stim_dur"
        )
    return float(crossings[0] * dt)


def classify_response(metrics: dict) -> str:
    n = metrics["n_spikes"]
    if n == 0:
        return "subthreshold"
    if n == 1:
        return "single_spike"
    return "bursting"