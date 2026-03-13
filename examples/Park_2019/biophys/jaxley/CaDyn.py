"""
Calcium concentration dynamics for Jaxley.

Tracks intracellular calcium concentration [Ca2+]_i in a thin submembrane shell,
coupling calcium channels (which produce I_Ca) to calcium-dependent channels
(which read [Ca2+]_i).

Physical model
--------------
A thin cylindrical shell of depth d beneath the membrane accumulates free
calcium from three processes:

  d[Ca2+]_i       gamma * I_Ca                  kt * [Ca2+]_i       [Ca2+]_inf - [Ca2+]_i
  --------- = - ----------------  (influx)  -  ---------------  +  ----------------------
     dt          2 * F * d * 1e-4              [Ca2+]_i + kd              tau_r

  - gamma : fraction of unbuffered calcium (0..1)
  - F     : Faraday's constant (96485.3 C/mol)
  - d     : shell depth in um (1e-4 converts to cm for unit consistency)
  - kt,kd : Michaelis-Menten pump parameters (kt=0 disables pump)
  - tau_r  : passive decay time constant (ms)
  - [Ca2+]_inf : resting calcium concentration (mM)

The factor 10000 arises from unit conversion:
  [mA/cm^2] / ([C/mol] * [um]) -> [mM/ms] requires x 1e4.

The negative sign ensures inward calcium current (I_Ca < 0 in the convention
where positive = outward) produces a positive concentration increase.
The max(0, ...) clamp prevents outward current from depleting calcium
faster than passive decay.

Integration
-----------
With kt=0 (Park_2019 configuration), the ODE is linear and can be rewritten as:

  d(cai)/dt = (cai_inf_eff - cai) / tau_r

where cai_inf_eff = [Ca2+]_inf + drive * tau_r. This is exactly the form
solved by Jaxley's exponential_euler integrator, giving the same numerical
accuracy as for gating variable kinetics.

Jaxley architecture note
------------------------
Jaxley stores the summed ionic current per ion type in the states dict
under the membrane current name (e.g. "i_ca"). This is populated by
_channel_currents before _step_channels is called, and passed to
update_states via the membrane_current_names mechanism. CaDyn reads
states["i_ca"] directly — the sum of all calcium channel currents
(CaHVA + CaLVA) — rather than redundantly reconstructing it from
gating variables.
"""

from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as jnp

FARADAY = 96485.3321  # C/mol


class CaDyn(Channel):
    """
    Intracellular calcium concentration dynamics for a thin submembrane shell.

    Reads the total calcium current (i_ca) from the shared Jaxley state dict,
    integrates the [Ca2+]_i ODE, and exposes the result as the 'CaCon_i'
    state for calcium-dependent channels (e.g. KCa).
    """

    def __init__(self, name="CaDyn"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "depth_CaDyn": 0.1,      # um: shell depth
            "taur_CaDyn": 50.0,      # ms: passive decay time constant
            "cainf_CaDyn": 1e-4,     # mM: resting [Ca2+]_i
            "gamma_CaDyn": 1.0,      # fraction of free (unbuffered) Ca2+
        }
        self.channel_states = {
            "CaCon_i": 1e-4,         # mM: initial [Ca2+]_i
        }
        self.current_name = "i_ca"

    def update_states(self, states, dt, v, params):
        cai = states["CaCon_i"]

        depth = params["depth_CaDyn"]
        taur = params["taur_CaDyn"]
        cainf = params["cainf_CaDyn"]
        gamma = params["gamma_CaDyn"]

        # Total calcium current from all Ca channels, summed by Jaxley
        # and passed via membrane_current_names.
        ica = states["i_ca"]

        # Current density -> concentration rate (mA/cm^2 -> mM/ms)
        drive = -(10000.0) * ica * gamma / (2.0 * FARADAY * depth)
        drive = jnp.maximum(drive, 0.0)

        # Rewrite ODE for exponential Euler:
        #   d(cai)/dt = drive + (cainf - cai) / taur
        #             = (cainf + drive * taur - cai) / taur
        cai_inf_eff = cainf + drive * taur
        new_cai = exponential_euler(cai, dt, cai_inf_eff, taur)

        return {"CaCon_i": new_cai}

    def compute_current(self, states, v, params):
        """CaDyn is a concentration mechanism — it produces no membrane current."""
        return 0.0

    def init_state(self, states, v, params, delta_t):
        """Initialize [Ca2+]_i to resting concentration."""
        return {"CaCon_i": params["cainf_CaDyn"]}
