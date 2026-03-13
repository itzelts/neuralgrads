# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class kap(Channel):
    """
    K-A channel from Klee Ficker and Heinemann
    """

    def __init__(self, name="kap"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_kap": 0.008,
            "vhalfn_kap": 11,
            "vhalfl_kap": -56,
            "a0l_kap": 0.05,
            "a0n_kap": 0.05,
            "zetan_kap": -1.5,
            "zetal_kap": 3,
            "gmn_kap": 0.55,
            "gml_kap": 1,
            "lmin_kap": 2,
            "nmin_kap": 0.1,
            "pw_kap": -1,
            "tq_kap": -40,
            "qq_kap": 5,
            "q10_kap": 5,
            "qtl_kap": 1
            }
        self.channel_states = {
            "n_kap": 0.0,
            "l_kap": 0.0
            }
        self._state_powers = {
            "n_kap": {'power': 1},
            "l_kap": {'power': 1}
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
        q10 = self.channel_params.get(f"q10_kap")
        reference_temp = self.channel_params.get(f"temp_kap")
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
        vhalfn = self.channel_params.get("vhalfn_kap", 1)
        zetan = self.channel_params.get("zetan_kap", 1)
        pw = self.channel_params.get("pw_kap", 1)
        tq = self.channel_params.get("tq_kap", 1)
        qq = self.channel_params.get("qq_kap", 1)
        
        zeta = zetan + (pw / (1 + np.exp(((v - tq) / qq))))
        alpn = np.exp(((((0.001 * zeta) * (v - vhalfn)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return alpn
    
    
    def betn(self, v):
        vhalfn = self.channel_params.get("vhalfn_kap", 1)
        zetan = self.channel_params.get("zetan_kap", 1)
        gmn = self.channel_params.get("gmn_kap", 1)
        pw = self.channel_params.get("pw_kap", 1)
        tq = self.channel_params.get("tq_kap", 1)
        qq = self.channel_params.get("qq_kap", 1)
        
        zeta = zetan + (pw / (1 + np.exp(((v - tq) / qq))))
        betn = np.exp((((((0.001 * zeta) * gmn) * (v - vhalfn)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return betn
    
    
    def alpl(self, v):
        vhalfl = self.channel_params.get("vhalfl_kap", 1)
        zetal = self.channel_params.get("zetal_kap", 1)
        
        alpl = np.exp(((((0.001 * zetal) * (v - vhalfl)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return alpl
    
    
    def betl(self, v):
        vhalfl = self.channel_params.get("vhalfl_kap", 1)
        zetal = self.channel_params.get("zetal_kap", 1)
        gml = self.channel_params.get("gml_kap", 1)
        
        betl = np.exp((((((0.001 * zetal) * gml) * (v - vhalfl)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return betl
    
    def compute_kinetic_variables(self, v):
        a0n = self.channel_params.get("a0n_kap", 1)
        lmin = self.channel_params.get("lmin_kap", 1)
        nmin = self.channel_params.get("nmin_kap", 1)
        q10 = self.channel_params.get("q10_kap", 1)
        qtl = self.channel_params.get("qtl_kap", 1)
        
        qt = q10 ** ((self.temperature - 24) / 10)
        a = self.alpn(v)
        nInf = 1 / (1 + a)
        nTau = self.betn(v) / ((qt * a0n) * (1 + a))
        conditions = [nTau < nmin, ~(nTau < nmin)]
        choices = [nmin, nTau]
        nTau = np.select(conditions, choices)
        a = self.alpl(v)
        lInf = 1 / (1 + a)
        lTau = (0.26 * (v + 50)) / qtl
        conditions = [lTau < (lmin / qtl), ~(lTau < (lmin / qtl))]
        choices = [lmin / qtl, lTau]
        lTau = np.select(conditions, choices)
        return nInf, nTau, lInf, lTau

    def update_states(self, states, dt, v, params):
        n = states['n_kap']
        l = states['l_kap']
        nInf, nTau, lInf, lTau = self.compute_kinetic_variables(v)
        new_n = exponential_euler(n, dt, nInf, nTau)
        new_l = exponential_euler(l, dt, lInf, lTau)
        return {
            "n_kap": new_n,
            "l_kap": new_l
            }

    def compute_current(self, states, v, params):
        n = states['n_kap']
        l = states['l_kap']
        gbar = params["gbar_kap"]
        E = params.get("E_k", -80)
        nInf, nTau, lInf, lTau = self.compute_kinetic_variables(v)
        g = self.tadj * gbar * n**1 * l**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        nInf, nTau, lInf, lTau = self.compute_kinetic_variables(v)
        return {
            "n_kap": nInf,
            "l_kap": lInf
            }

