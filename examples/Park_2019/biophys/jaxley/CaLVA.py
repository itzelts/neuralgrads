# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class CaLVA(Channel):
    """
    T-type Ca channel
    """

    def __init__(self, name="CaLVA"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_CaLVA": 0.0,
            "v12m_CaLVA": 50,
            "v12h_CaLVA": 78,
            "vwm_CaLVA": 7.4,
            "vwh_CaLVA": 5.0,
            "am_CaLVA": 3,
            "ah_CaLVA": 85,
            "vm1_CaLVA": 25,
            "vm2_CaLVA": 100,
            "vh1_CaLVA": 46,
            "vh2_CaLVA": 405,
            "wm1_CaLVA": 20,
            "wm2_CaLVA": 15,
            "wh1_CaLVA": 4,
            "wh2_CaLVA": 50
            }
        self.channel_states = {
            "m_CaLVA": 0.0,
            "h_CaLVA": 0.0
            }
        self._state_powers = {
            "m_CaLVA": {'power': 2},
            "h_CaLVA": {'power': 1}
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
        q10 = self.channel_params.get(f"q10_CaLVA")
        reference_temp = self.channel_params.get(f"temp_CaLVA")
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
        v12m = self.channel_params.get("v12m_CaLVA", 1)
        v12h = self.channel_params.get("v12h_CaLVA", 1)
        vwm = self.channel_params.get("vwm_CaLVA", 1)
        vwh = self.channel_params.get("vwh_CaLVA", 1)
        am = self.channel_params.get("am_CaLVA", 1)
        ah = self.channel_params.get("ah_CaLVA", 1)
        vm1 = self.channel_params.get("vm1_CaLVA", 1)
        vm2 = self.channel_params.get("vm2_CaLVA", 1)
        vh1 = self.channel_params.get("vh1_CaLVA", 1)
        vh2 = self.channel_params.get("vh2_CaLVA", 1)
        wm1 = self.channel_params.get("wm1_CaLVA", 1)
        wm2 = self.channel_params.get("wm2_CaLVA", 1)
        wh1 = self.channel_params.get("wh1_CaLVA", 1)
        wh2 = self.channel_params.get("wh2_CaLVA", 1)
        
        mInf = 1.0 / (1 + np.exp((-(v + v12m) / vwm)))
        hInf = 1.0 / (1 + np.exp(((v + v12h) / vwh)))
        mTau = am + (1.0 / (np.exp(((v + vm1) / wm1)) + np.exp((-(v + vm2) / wm2))))
        hTau = ah + (1.0 / (np.exp(((v + vh1) / wh1)) + np.exp((-(v + vh2) / wh2))))
        return mInf, mTau, hInf, hTau

    def update_states(self, states, dt, v, params):
        m = states['m_CaLVA']
        h = states['h_CaLVA']
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        new_m = exponential_euler(m, dt, mInf, mTau)
        new_h = exponential_euler(h, dt, hInf, hTau)
        return {
            "m_CaLVA": new_m,
            "h_CaLVA": new_h
            }

    def compute_current(self, states, v, params):
        m = states['m_CaLVA']
        h = states['h_CaLVA']
        gbar = params["gbar_CaLVA"]
        E = params.get("E_ca", 140)
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        g = self.tadj * gbar * m**2 * h**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        return {
            "m_CaLVA": mInf,
            "h_CaLVA": hInf
            }

