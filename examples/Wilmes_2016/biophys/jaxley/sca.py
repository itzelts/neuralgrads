# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class sca(Channel):
    """
    
    """

    def __init__(self, name="sca"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_sca": 0.0,
            "vshift_sca": 0,
            "actF_sca": 1,
            "inactF_sca": 3,
            "q10_sca": 2.3,
            "temp_sca": 23
            }
        self.channel_states = {
            "m_sca": 0.0,
            "h_sca": 0.0
            }
        self._state_powers = {
            "m_sca": {'power': 2},
            "h_sca": {'power': 1}
            }
        self.ion = "ca"
        self.current_name = "i_ca"

        self.independent_var_name = "v"
        self.tadj = 1
        self.temperature = 24

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
        self.temperature = temperature
        q10 = self.channel_params.get(f"q10_sca")
        reference_temp = self.channel_params.get(f"temp_sca")
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
        actF = self.channel_params.get("actF_sca", 1)
        inactF = self.channel_params.get("inactF_sca", 1)
        q10 = self.channel_params.get("q10_sca", 1)
        temp = self.channel_params.get("temp_sca", 1)
        
        qt = q10 ** ((self.temperature - temp) / 10)
        a = ((0.055 * (-27 - v)) / (np.exp(((-27 - v) / 3.8)) - 1)) / actF
        b = (0.94 * np.exp(((-75 - v) / 17))) / actF
        mTau = (1 / (a + b)) / qt
        mInf = a / (a + b)
        a = (0.000457 * np.exp(((-13 - v) / 50))) / inactF
        b = (0.0065 / (np.exp(((-v - 15) / 28)) + 1)) / inactF
        hTau = (1 / (a + b)) / qt
        hInf = a / (a + b)
        return mInf, mTau, hInf, hTau

    def update_states(self, states, dt, v, params):
        m = states['m_sca']
        h = states['h_sca']
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        new_m = exponential_euler(m, dt, mInf, mTau)
        new_h = exponential_euler(h, dt, hInf, hTau)
        return {
            "m_sca": new_m,
            "h_sca": new_h
            }

    def compute_current(self, states, v, params):
        m = states['m_sca']
        h = states['h_sca']
        gbar = params["gbar_sca"]
        E = params.get("E_ca", 140)
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        g = self.tadj * gbar * m**2 * h**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        return {
            "m_sca": mInf,
            "h_sca": hInf
            }

