# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class it2(Channel):
    """
    
    """

    def __init__(self, name="it2"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_it2": 0.0,
            "vshift_it2": 0,
            "v12m_it2": 50,
            "v12h_it2": 78,
            "vwm_it2": 7.4,
            "vwh_it2": 5.0,
            "am_it2": 3,
            "ah_it2": 85,
            "vm1_it2": 25,
            "vm2_it2": 100,
            "vh1_it2": 46,
            "vh2_it2": 405,
            "wm1_it2": 20,
            "wm2_it2": 15,
            "wh1_it2": 4,
            "wh2_it2": 50
            }
        self.channel_states = {
            "m_it2": 0.0,
            "h_it2": 0.0
            }
        self._state_powers = {
            "m_it2": {'power': 2},
            "h_it2": {'power': 1}
            }
        self.ion = "ca"
        self.current_name = "i_ca"

        self.independent_var_name = "v"
        self.tadj = 1

    def set_tadj(self, temperature):
        """
        Set the temperature adjustment factor for the channel kinetics.

        Parameters
        ----------
        temperature : float
            The temperature in degrees Celsius.

        Notes
        -----
        The temperature adjustment factor is calculated as:
        tadj = q10 ** ((temperature - reference_temp) / 10)
        where q10 is the temperature coefficient and reference_temp is the
        temperature at which the channel kinetics were measured.
        """
        q10 = self.channel_params.get(f"q10_it2")
        reference_temp = self.channel_params.get(f"temp_it2")
        if q10 is None or reference_temp is None:
            self.tadj = 1
            print(f"Warning: q10 or reference temperature not set for {self.name}. Using default tadj = 1.")
        else:
            self.tadj = q10 ** ((temperature - reference_temp) / 10)

    def __getitem__(self, item):
        return self.channel_params[item]

    def __setitem__(self, item, value):
        self.channel_params[item] = value
        
    
    def compute_kinetic_variables(self, v):
        v = v + self.channel_params.get("vshift_it2", 0)
        v12m = self.channel_params.get("v12m_it2", 1)
        v12h = self.channel_params.get("v12h_it2", 1)
        vwm = self.channel_params.get("vwm_it2", 1)
        vwh = self.channel_params.get("vwh_it2", 1)
        am = self.channel_params.get("am_it2", 1)
        ah = self.channel_params.get("ah_it2", 1)
        vm1 = self.channel_params.get("vm1_it2", 1)
        vm2 = self.channel_params.get("vm2_it2", 1)
        vh1 = self.channel_params.get("vh1_it2", 1)
        vh2 = self.channel_params.get("vh2_it2", 1)
        wm1 = self.channel_params.get("wm1_it2", 1)
        wm2 = self.channel_params.get("wm2_it2", 1)
        wh1 = self.channel_params.get("wh1_it2", 1)
        wh2 = self.channel_params.get("wh2_it2", 1)
        
        mInf = 1.0 / (1 + np.exp((-(v + v12m) / vwm)))
        hInf = 1.0 / (1 + np.exp(((v + v12h) / vwh)))
        mTau = am + (1.0 / (np.exp(((v + vm1) / wm1)) + np.exp((-(v + vm2) / wm2))))
        hTau = ah + (1.0 / (np.exp(((v + vh1) / wh1)) + np.exp((-(v + vh2) / wh2))))
        return mInf, mTau, hInf, hTau

    def update_states(self, states, dt, v, params):
        m = states['m_it2']
        h = states['h_it2']
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        new_m = exponential_euler(m, dt, mInf, mTau)
        new_h = exponential_euler(h, dt, hInf, hTau)
        return {
            "m_it2": new_m,
            "h_it2": new_h
            }

    def compute_current(self, states, v, params):
        m = states['m_it2']
        h = states['h_it2']
        gbar = params["gbar_it2"]
        E = params.get("E_ca", 140)
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        g = self.tadj * gbar * m**2 * h**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        return {
            "m_it2": mInf,
            "h_it2": hInf
            }

