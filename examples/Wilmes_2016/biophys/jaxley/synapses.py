"""Destexhe et al. (1998) synapse models for Jaxley.

Reference:
    Destexhe, A., Mainen, Z. F., & Bhalla, U. S. (1998).
    Kinetic models of synaptic transmission. In *Methods in Neuronal Modeling*
    (2nd ed., Ch. 1). C. Koch & I. Segev, Eds. MIT Press.
    https://www.csc.kth.se/utbildning/kth/kurser/DD2435/biomod12/kursbunt/f9/KochCh1Destexhe.pdf

ModelDB code: https://modeldb.science/18500?tab=2
"""

from typing import Dict, Optional

import jax.numpy as jnp
from jax.lax import select
from jaxley.solver_gate import save_exp
from jaxley.synapses import Synapse

__all__ = ["AMPA", "GABAa", "GABAb", "NMDA", "exptable"]

META = {
    "reference": "Destexhe, et al. (1998).",
    "doi": "https://www.csc.kth.se/utbildning/kth/kurser/DD2435/biomod12/kursbunt/f9/KochCh1Destexhe.pdf",
    "code": "https://modeldb.science/18500?tab=2",
    "note": "no doi for this book chapter, link to the pdf instead",
    "species": "unknown",
    "cell_type": "unknown",
}


def exptable(x):
    """Approximate exponential function used in NEURON's AMPA model."""
    return select((x > -10) & (x < 10), save_exp(x), jnp.zeros_like(x))


class AMPA(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gAMPA": 0.1e-3,
            f"{name}_eAMPA": 0.0,
            f"{name}_Cmax": 1,
            f"{name}_Cdur": 1,
            f"{name}_alpha": 1.1,
            f"{name}_beta": 0.19,
            f"{name}_vt_pre": 0,
            f"{name}_deadtime": 1,
        }
        self.synapse_states = {
            f"{name}_R": 0,
            f"{name}_C": 0,
            f"{name}_lastrelease": -1000,
            f"{name}_timecount": -1,
            f"{name}_R0": 0,
            f"{name}_R1": 0,
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        name = self.name
        timecount = u[f"{name}_timecount"]
        new_timecount = select(
            timecount == -1,
            params[f"{name}_Cdur"],
            timecount - delta_t,
        )

        new_release_condition = (pre_voltage > params[f"{name}_vt_pre"]) & (
            new_timecount <= -params[f"{name}_deadtime"]
        )
        Cmax = params[f"{name}_Cmax"]
        Cdur = params[f"{name}_Cdur"]
        C = u[f"{name}_C"]
        new_C = select(
            new_release_condition,
            Cmax,
            select(new_timecount > 0, C, jnp.zeros_like(C)),
        )

        new_lastrelease = select(
            new_release_condition,
            jnp.zeros_like(u[f"{name}_lastrelease"]),
            u[f"{name}_lastrelease"] + delta_t,
        )

        alpha = params[f"{name}_alpha"]
        beta = params[f"{name}_beta"]
        R_inf = Cmax * alpha / (Cmax * alpha + beta)
        R_tau = 1 / (alpha * Cmax + beta)

        R0 = u[f"{name}_R0"]
        R1 = u[f"{name}_R1"]
        R = u[f"{name}_R"]
        new_R0 = select(new_release_condition, R, R0)
        new_R1 = select(C > 0, R, R1)

        time_since_release = new_lastrelease - Cdur
        new_R = select(
            new_C > 0,
            R_inf + (new_R0 - R_inf) * exptable(-(new_lastrelease) / R_tau),
            new_R1 * exptable(-time_since_release * beta),
        )

        new_timecount = select(new_release_condition, Cdur, new_timecount)

        return {
            f"{name}_R": new_R,
            f"{name}_C": new_C,
            f"{name}_lastrelease": new_lastrelease,
            f"{name}_timecount": new_timecount,
            f"{name}_R0": new_R0,
            f"{name}_R1": new_R1,
        }

    def init_state(self, v, params):
        name = self.name
        return {
            f"{name}_R": 0,
            f"{name}_R0": 0,
            f"{name}_R1": 0,
            f"{name}_C": 0,
            f"{name}_lastrelease": -1000,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        name = self.name
        g_syn = params[f"{name}_gAMPA"] * u[f"{name}_R"]
        return g_syn * (post_voltage - params[f"{name}_eAMPA"])


class GABAa(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gGABAa": 0.1e-3,
            f"{name}_eGABAa": -80.0,
            f"{name}_Cmax": 1,
            f"{name}_Cdur": 1,
            f"{name}_alpha": 5,
            f"{name}_beta": 0.18,
            f"{name}_vt_pre": 0,
            f"{name}_deadtime": 1,
        }
        self.synapse_states = {
            f"{name}_R": 0,
            f"{name}_C": 0,
            f"{name}_lastrelease": -1000,
            f"{name}_timecount": -1,
            f"{name}_R0": 0,
            f"{name}_R1": 0,
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        name = self._name
        timecount = u[f"{name}_timecount"]
        new_timecount = select(
            timecount == -1,
            params[f"{name}_Cdur"],
            timecount - delta_t,
        )

        new_release_condition = (pre_voltage > params[f"{name}_vt_pre"]) & (
            new_timecount <= -params[f"{name}_deadtime"]
        )
        Cmax = params[f"{name}_Cmax"]
        Cdur = params[f"{name}_Cdur"]
        C = u[f"{name}_C"]
        new_C = select(
            new_release_condition,
            Cmax,
            select(new_timecount > 0, C, jnp.zeros_like(C)),
        )

        new_lastrelease = select(
            new_release_condition,
            jnp.zeros_like(u[f"{name}_lastrelease"]),
            u[f"{name}_lastrelease"] + delta_t,
        )

        alpha = params[f"{name}_alpha"]
        beta = params[f"{name}_beta"]
        R_inf = Cmax * alpha / (Cmax * alpha + beta)
        R_tau = 1 / (alpha * Cmax + beta)

        R0 = u[f"{name}_R0"]
        R1 = u[f"{name}_R1"]
        R = u[f"{name}_R"]
        new_R0 = select(new_release_condition, R, R0)
        new_R1 = select(C > 0, R, R1)

        time_since_release = new_lastrelease - Cdur
        new_R = select(
            new_C > 0,
            R_inf + (new_R0 - R_inf) * exptable(-(new_lastrelease) / R_tau),
            new_R1 * exptable(-beta * time_since_release),
        )

        new_timecount = select(new_release_condition, Cdur, new_timecount)

        return {
            f"{name}_R": new_R,
            f"{name}_C": new_C,
            f"{name}_lastrelease": new_lastrelease,
            f"{name}_timecount": new_timecount,
            f"{name}_R0": new_R0,
            f"{name}_R1": new_R1,
        }

    def init_state(self, v, params):
        name = self._name
        return {
            f"{name}_R": 0,
            f"{name}_R0": 0,
            f"{name}_R1": 0,
            f"{name}_C": 0,
            f"{name}_lastrelease": -1000,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        name = self._name
        g_syn = params[f"{name}_gGABAa"] * u[f"{name}_R"]
        return g_syn * (post_voltage - params[f"{name}_eGABAa"])


class GABAb(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gGABAb": 0.1e-3,
            f"{name}_eGABAb": -95.0,
            f"{name}_Cmax": 0.5,
            f"{name}_Cdur": 0.3,
            f"{name}_vt_pre": 0,
            f"{name}_deadtime": 1,
            f"{name}_K1": 0.09,
            f"{name}_K2": 1.2e-3,
            f"{name}_K3": 180e-3,
            f"{name}_K4": 34e-3,
            f"{name}_KD": 100,
            f"{name}_n": 4,
        }
        self.synapse_states = {
            f"{name}_R": 0,
            f"{name}_G": 0,
            f"{name}_C": 0,
            f"{name}_lastrelease": -1000,
            f"{name}_timecount": -1,
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        name = self._name
        timecount = u[f"{name}_timecount"]
        new_timecount = select(
            timecount == -1,
            params[f"{name}_Cdur"],
            timecount - delta_t,
        )

        new_release_condition = (pre_voltage > params[f"{name}_vt_pre"]) & (
            new_timecount <= -params[f"{name}_deadtime"]
        )
        Cmax = params[f"{name}_Cmax"]
        Cdur = params[f"{name}_Cdur"]
        C = u[f"{name}_C"]
        new_C = select(
            new_release_condition,
            Cmax,
            select(new_timecount > 0, C, jnp.zeros_like(C)),
        )

        new_lastrelease = select(
            new_release_condition,
            jnp.zeros_like(u[f"{name}_lastrelease"]),
            u[f"{name}_lastrelease"] + delta_t,
        )

        R = u[f"{name}_R"]
        G = u[f"{name}_G"]
        K1 = params[f"{name}_K1"]
        K2 = params[f"{name}_K2"]
        K3 = params[f"{name}_K3"]
        K4 = params[f"{name}_K4"]
        new_R = R + delta_t * (K1 * C * (1 - R) - K2 * R)
        new_G = G + delta_t * (K3 * R - K4 * G)

        new_timecount = select(new_release_condition, Cdur, new_timecount)

        return {
            f"{name}_R": new_R,
            f"{name}_G": new_G,
            f"{name}_C": new_C,
            f"{name}_lastrelease": new_lastrelease,
            f"{name}_timecount": new_timecount,
        }

    def init_state(self, v, params):
        name = self._name
        return {
            f"{name}_R": 0,
            f"{name}_G": 0,
            f"{name}_lastrelease": -1000,
            f"{name}_C": 0,
            f"{name}_timecount": -1,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        name = self._name
        KD = params[f"{name}_KD"]
        n = params[f"{name}_n"]
        Gn = u[f"{name}_G"] ** n
        g_syn = params[f"{name}_gGABAb"] * Gn / (Gn + KD)
        return g_syn * (post_voltage - params[f"{name}_eGABAb"])


class NMDA(Synapse):
    def __init__(self, name: Optional[str] = None):
        self._name = name = name if name else self.__class__.__name__

        self.synapse_params = {
            f"{name}_gNMDA": 0.1e-3,
            f"{name}_eNMDA": 0.0,
            f"{name}_Cmax": 1,
            f"{name}_Cdur": 1,
            f"{name}_alpha": 0.072,
            f"{name}_beta": 0.0066,
            f"{name}_vt_pre": 0,
            f"{name}_deadtime": 1,
            f"{name}_mg": 1,
        }
        self.synapse_states = {
            f"{name}_R": 0,
            f"{name}_C": 0,
            f"{name}_lastrelease": -1000,
            f"{name}_timecount": -1,
            f"{name}_R0": 0,
            f"{name}_R1": 0,
        }
        self.META = META

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        name = self._name
        timecount = u[f"{name}_timecount"]
        new_timecount = select(
            timecount == -1,
            params[f"{name}_Cdur"],
            timecount - delta_t,
        )

        new_release_condition = (pre_voltage > params[f"{name}_vt_pre"]) & (
            new_timecount <= -params[f"{name}_deadtime"]
        )
        Cmax = params[f"{name}_Cmax"]
        Cdur = params[f"{name}_Cdur"]
        C = u[f"{name}_C"]
        new_C = select(
            new_release_condition,
            Cmax,
            select(new_timecount > 0, C, jnp.zeros_like(C)),
        )

        new_lastrelease = select(
            new_release_condition,
            jnp.zeros_like(u[f"{name}_lastrelease"]),
            u[f"{name}_lastrelease"] + delta_t,
        )

        R0 = u[f"{name}_R0"]
        R1 = u[f"{name}_R1"]
        R = u[f"{name}_R"]
        alpha = params[f"{name}_alpha"]
        beta = params[f"{name}_beta"]
        Rinf = Cmax * alpha / (Cmax * alpha + beta)
        Rtau = 1 / (alpha * Cmax + beta)
        new_R0 = select(new_release_condition, R, R0)
        new_R1 = select(C > 0, R, R1)
        time_since_release = new_lastrelease - Cdur

        new_R = select(
            new_C > 0,
            Rinf + (new_R0 - Rinf) * exptable(-(new_lastrelease) / Rtau),
            new_R1 * exptable(-beta * time_since_release),
        )

        new_timecount = select(new_release_condition, Cdur, new_timecount)

        return {
            f"{name}_R": new_R,
            f"{name}_C": new_C,
            f"{name}_lastrelease": new_lastrelease,
            f"{name}_timecount": new_timecount,
            f"{name}_R0": new_R0,
            f"{name}_R1": new_R1,
        }

    def init_state(self, v, params):
        name = self._name
        return {
            f"{name}_R": 0,
            f"{name}_R0": 0,
            f"{name}_R1": 0,
            f"{name}_C": 0,
            f"{name}_lastrelease": -1000,
            f"{name}_timecount": -1,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        name = self._name
        R = u[f"{name}_R"]
        B = self.mgblock(post_voltage, params[f"{name}_mg"])
        g_syn = params[f"{name}_gNMDA"] * R * B
        return g_syn * (post_voltage - params[f"{name}_eNMDA"])

    @staticmethod
    def mgblock(v, mg_concentration):
        """Voltage-dependent magnesium block factor."""
        return 1 / (1 + save_exp(0.062 * (-v)) * (mg_concentration / 3.57))
