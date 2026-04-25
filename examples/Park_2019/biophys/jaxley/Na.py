# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class Na(Channel):
    """
    Na channel
    """

    def __init__(self, name="Na"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_Na": 0.0,
            "Rma_Na": 0.182,
            "Rmb_Na": 0.14,
            "v12m_Na": -30,
            "qm_Na": 9.8,
            "Rhb_Na": 0.0091,
            "Rha_Na": 0.024,
            "v12ha_Na": -45,
            "v12hb_Na": -70,
            "qh_Na": 5,
            "v12hinf_Na": -60,
            "qhinf_Na": 6.2,
            "temp_Na": 23,
            "q10_Na": 2.3
            }
        self.channel_states = {
            "m_Na": 0.0,
            "h_Na": 0.0
            }
        self._state_powers = {
            "m_Na": {'power': 3},
            "h_Na": {'power': 1}
            }
        self.ion = "na"
        self.current_name = "i_na"

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
        q10 = self.channel_params.get(f"q10_Na")
        reference_temp = self.channel_params.get(f"temp_Na")
        if q10 is None or reference_temp is None:
            self.tadj = 1
            print(f"Warning: q10 or reference temperature not set for {self.name}. Using default tadj = 1.")
        else:
            self.tadj = q10 ** ((temperature - reference_temp) / 10)

    def __getitem__(self, item):
        return self.channel_params[item]

    def __setitem__(self, item, value):
        self.channel_params[item] = value
        
    
    def rateconst2(self, v, r, v12, q):
        conditions = [np.abs(((v - v12) / q)) > 1e-06, ~(np.abs(((v - v12) / q)) > 1e-06)]
        choices = [(r * (v - v12)) / (1 - np.exp((-(v - v12) / q))), r * q]
        rateconst2 = np.select(conditions, choices)
        return rateconst2
    
    
    def rateconst(self, v, r, v12, q):
        conditions = [np.abs(((v - v12) / q)) > 1e-06, ~(np.abs(((v - v12) / q)) > 1e-06)]
        choices = [(r * (v - v12)) / (1 - np.exp((-(v - v12) / q))), r * q]
        rateconst = np.select(conditions, choices)
        return rateconst
    
    def compute_kinetic_variables(self, v):
        Rma = self.channel_params.get("Rma_Na", 1)
        Rmb = self.channel_params.get("Rmb_Na", 1)
        v12m = self.channel_params.get("v12m_Na", 1)
        qm = self.channel_params.get("qm_Na", 1)
        Rhb = self.channel_params.get("Rhb_Na", 1)
        Rha = self.channel_params.get("Rha_Na", 1)
        v12ha = self.channel_params.get("v12ha_Na", 1)
        v12hb = self.channel_params.get("v12hb_Na", 1)
        qh = self.channel_params.get("qh_Na", 1)
        v12hinf = self.channel_params.get("v12hinf_Na", 1)
        qhinf = self.channel_params.get("qhinf_Na", 1)
        
        alpm = self.rateconst(v, Rma, v12m, qm)
        betm = self.rateconst(-v, Rmb, -v12m, qm)
        alph = self.rateconst(v, Rha, v12ha, qh)
        beth = self.rateconst(-v, Rhb, -v12hb, qh)
        mTau = 1 / (self.tadj * (alpm + betm))
        mInf = alpm / (alpm + betm)
        hTau = 1 / (self.tadj * (alph + beth))
        hInf = 1 / (1 + np.exp(((v - v12hinf) / qhinf)))
        return mInf, mTau, hInf, hTau

    def update_states(self, states, dt, v, params):
        m = states['m_Na']
        h = states['h_Na']
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        new_m = exponential_euler(m, dt, mInf, mTau)
        new_h = exponential_euler(h, dt, hInf, hTau)
        return {
            "m_Na": new_m,
            "h_Na": new_h
            }

    def compute_current(self, states, v, params):
        m = states['m_Na']
        h = states['h_Na']
        gbar = params["gbar_Na"]
        E = params.get("E_na", 60)
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        g = self.tadj * gbar * m**3 * h**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        return {
            "m_Na": mInf,
            "h_Na": hInf
            }

