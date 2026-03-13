# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class Km(Channel):
    """
    V-g K+ channel (Muscarinic or M-Type?)
    """

    def __init__(self, name="Km"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_Km": 0.0,
            "Ra_Km": 0.001,
            "Rb_Km": 0.001,
            "v12_Km": -30,
            "q_Km": 9,
            "temp_Km": 23,
            "q10_Km": 2.3
            }
        self.channel_states = {
            "n_Km": 0.0
            }
        self._state_powers = {
            "n_Km": {'power': 1}
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
        q10 = self.channel_params.get(f"q10_Km")
        reference_temp = self.channel_params.get(f"temp_Km")
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
        Ra = self.channel_params.get("Ra_Km", 1)
        Rb = self.channel_params.get("Rb_Km", 1)
        v12 = self.channel_params.get("v12_Km", 1)
        q = self.channel_params.get("q_Km", 1)
        
        alpn = self.rateconst(v, Ra, v12, q)
        betn = self.rateconst(v, -Rb, v12, -q)
        nTau = (1 / self.tadj) / (alpn + betn)
        nInf = alpn / (alpn + betn)
        return nInf, nTau

    def update_states(self, states, dt, v, params):
        n = states['n_Km']
        nInf, nTau = self.compute_kinetic_variables(v)
        new_n = exponential_euler(n, dt, nInf, nTau)
        return {
            "n_Km": new_n
            }

    def compute_current(self, states, v, params):
        n = states['n_Km']
        gbar = params["gbar_Km"]
        E = params.get("E_k", -80)
        nInf, nTau = self.compute_kinetic_variables(v)
        g = self.tadj * gbar * n**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        nInf, nTau = self.compute_kinetic_variables(v)
        return {
            "n_Km": nInf
            }

