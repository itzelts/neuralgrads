# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class kdrca1(Channel):
    """
    K-DR channel
    """

    def __init__(self, name="kdrca1"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_kdrca1": 0.003,
            "vhalfn_kdrca1": 13,
            "a0n_kdrca1": 0.02,
            "zetan_kdrca1": -3,
            "gmn_kdrca1": 0.7,
            "nmax_kdrca1": 2,
            "q10_kdrca1": 1
            }
        self.channel_states = {
            "n_kdrca1": 0.0
            }
        self._state_powers = {
            "n_kdrca1": {'power': 1}
            }
        self.ion = "k"
        self.current_name = "i_k"

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
        q10 = self.channel_params.get(f"q10_kdrca1")
        reference_temp = self.channel_params.get(f"temp_kdrca1")
        if q10 is None or reference_temp is None:
            self.tadj = 1
            print(f"Warning: q10 or reference temperature not set for {self.name}. Using default tadj = 1.")
        else:
            self.tadj = q10 ** ((temperature - reference_temp) / 10)

    def __getitem__(self, item):
        return self.channel_params[item]

    def __setitem__(self, item, value):
        self.channel_params[item] = value
        
    
    def alpn(self, v):
        vhalfn = self.channel_params.get("vhalfn_kdrca1", 1)
        zetan = self.channel_params.get("zetan_kdrca1", 1)
        
        alpn = np.exp(((((0.001 * zetan) * (v - vhalfn)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return alpn
    
    
    def betn(self, v):
        vhalfn = self.channel_params.get("vhalfn_kdrca1", 1)
        zetan = self.channel_params.get("zetan_kdrca1", 1)
        gmn = self.channel_params.get("gmn_kdrca1", 1)
        
        betn = np.exp((((((0.001 * zetan) * gmn) * (v - vhalfn)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return betn
    
    def compute_kinetic_variables(self, v):
        a0n = self.channel_params.get("a0n_kdrca1", 1)
        nmax = self.channel_params.get("nmax_kdrca1", 1)
        q10 = self.channel_params.get("q10_kdrca1", 1)
        
        qt = q10 ** ((self.temperature - 24) / 10)
        a = self.alpn(v)
        nInf = 1 / (1 + a)
        nTau = self.betn(v) / ((qt * a0n) * (1 + a))
        conditions = [nTau < nmax, ~(nTau < nmax)]
        choices = [nmax, nTau]
        nTau = np.select(conditions, choices)
        return nInf, nTau

    def update_states(self, states, dt, v, params):
        n = states['n_kdrca1']
        nInf, nTau = self.compute_kinetic_variables(v)
        new_n = exponential_euler(n, dt, nInf, nTau)
        return {
            "n_kdrca1": new_n
            }

    def compute_current(self, states, v, params):
        n = states['n_kdrca1']
        gbar = params["gbar_kdrca1"]
        E = params.get("E_k", -80)
        nInf, nTau = self.compute_kinetic_variables(v)
        g = self.tadj * gbar * n**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        nInf, nTau = self.compute_kinetic_variables(v)
        return {
            "n_kdrca1": nInf
            }

