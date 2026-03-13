# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class Kv(Channel):
    """
    Kv_Park_ref
    """

    def __init__(self, name="Kv"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_Kv": 0.0,
            "Ra_Kv": 0.02,
            "Rb_Kv": 0.006,
            "v12_Kv": 25,
            "q_Kv": 9,
            "temp_Kv": 23,
            "q10_Kv": 2.3
            }
        self.channel_states = {
            "n_Kv": 0.0
            }
        self._state_powers = {
            "n_Kv": {'power': 1}
            }
        self.ion = "k"
        self.current_name = "i_k"

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
        q10 = self.channel_params.get(f"q10_Kv")
        reference_temp = self.channel_params.get(f"temp_Kv")
        if q10 is None or reference_temp is None:
            self.tadj = 1
            print(f"Warning: q10 or reference temperature not set for {self.name}. Using default tadj = 1.")
        else:
            self.tadj = q10 ** ((temperature - reference_temp) / 10)

    def __getitem__(self, item):
        return self.channel_params[item]

    def __setitem__(self, item, value):
        self.channel_params[item] = value
        
    
    def rateconst(self, v, r, th, q):
        rateconst = (r * (v - th)) / (1 - np.exp((-(v - th) / q)))
        return rateconst
    
    def compute_kinetic_variables(self, v):
        Ra = self.channel_params.get("Ra_Kv", 1)
        Rb = self.channel_params.get("Rb_Kv", 1)
        v12 = self.channel_params.get("v12_Kv", 1)
        q = self.channel_params.get("q_Kv", 1)
        
        alpn = self.rateconst(v, Ra, v12, q)
        betn = self.rateconst(v, -Rb, v12, -q)
        nTau = 1 / (self.tadj * (alpn + betn))
        nInf = alpn / (alpn + betn)
        return nInf, nTau

    def update_states(self, states, dt, v, params):
        n = states['n_Kv']
        nInf, nTau = self.compute_kinetic_variables(v)
        new_n = exponential_euler(n, dt, nInf, nTau)
        return {
            "n_Kv": new_n
            }

    def compute_current(self, states, v, params):
        n = states['n_Kv']
        gbar = params["gbar_Kv"]
        E = params.get("E_k", -80)
        nInf, nTau = self.compute_kinetic_variables(v)
        g = self.tadj * gbar * n**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        nInf, nTau = self.compute_kinetic_variables(v)
        return {
            "n_Kv": nInf
            }

