# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class kca(Channel):
    """
    
    """

    def __init__(self, name="kca"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_kca": 0.0,
            "caix_kca": 1,
            "Ra_kca": 0.01,
            "Rb_kca": 0.02,
            "q10_kca": 2.3,
            "temp_kca": 23
            }
        self.channel_states = {
            "n_kca": 0.0,
            "CaCon_i": 5e-05
            }
        self._state_powers = {
            "n_kca": {'power': 1}
            }
        self.ion = "k"
        self.current_name = "i_k"

        self.independent_var_name = "cai"
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
        q10 = self.channel_params.get(f"q10_kca")
        reference_temp = self.channel_params.get(f"temp_kca")
        if q10 is None or reference_temp is None:
            self.tadj = 1
            print(f"Warning: q10 or reference temperature not set for {self.name}. Using default tadj = 1.")
        else:
            self.tadj = q10 ** ((temperature - reference_temp) / 10)

    def __getitem__(self, item):
        return self.channel_params[item]

    def __setitem__(self, item, value):
        self.channel_params[item] = value
        
    
    def compute_kinetic_variables(self, cai):
        caix = self.channel_params.get("caix_kca", 1)
        Ra = self.channel_params.get("Ra_kca", 1)
        Rb = self.channel_params.get("Rb_kca", 1)
        q10 = self.channel_params.get("q10_kca", 1)
        temp = self.channel_params.get("temp_kca", 1)
        
        qt = q10 ** ((self.temperature - temp) / 10)
        a = Ra * (cai ** caix)
        nTau = (1 / (a + Rb)) / qt
        nInf = a / (a + Rb)
        return nInf, nTau

    def update_states(self, states, dt, v, params):
        n = states['n_kca']
        cai = states["CaCon_i"]
        nInf, nTau = self.compute_kinetic_variables(cai)
        new_n = exponential_euler(n, dt, nInf, nTau)
        return {
            "n_kca": new_n
            }

    def compute_current(self, states, v, params):
        n = states['n_kca']
        gbar = params["gbar_kca"]
        cai = states["CaCon_i"]
        E = params.get("E_k", -80)
        nInf, nTau = self.compute_kinetic_variables(cai)
        g = self.tadj * gbar * n**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        cai = states["CaCon_i"]
        nInf, nTau = self.compute_kinetic_variables(cai)
        return {
            "n_kca": nInf
            }

