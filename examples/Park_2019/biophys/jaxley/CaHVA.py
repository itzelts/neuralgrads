# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class CaHVA(Channel):
    """
    HVA Ca current
    """

    def __init__(self, name="CaHVA"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_CaHVA": 0.0,
            "Rma_CaHVA": 0.5,
            "Rmb_CaHVA": 0.1,
            "v12ma_CaHVA": -27,
            "v12mb_CaHVA": -75,
            "qma_CaHVA": 3.8,
            "qmb_CaHVA": 17,
            "Rha_CaHVA": 0.000457,
            "Rhb_CaHVA": 0.0065,
            "v12ha_CaHVA": -13,
            "v12hb_CaHVA": -15,
            "qha_CaHVA": 50,
            "qhb_CaHVA": 28,
            "temp_CaHVA": 23,
            "q10_CaHVA": 2.3
            }
        self.channel_states = {
            "m_CaHVA": 0.0,
            "h_CaHVA": 0.0
            }
        self._state_powers = {
            "m_CaHVA": {'power': 2},
            "h_CaHVA": {'power': 1}
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
        q10 = self.channel_params.get(f"q10_CaHVA")
        reference_temp = self.channel_params.get(f"temp_CaHVA")
        if q10 is None or reference_temp is None:
            self.tadj = 1
            print(f"Warning: q10 or reference temperature not set for {self.name}. Using default tadj = 1.")
        else:
            self.tadj = q10 ** ((temperature - reference_temp) / 10)

    def __getitem__(self, item):
        return self.channel_params[item]

    def __setitem__(self, item, value):
        self.channel_params[item] = value
        
    
    def f_lexp(self, v, R, v12, q):
        dv = -(v - v12)
        f_lexp = (R * dv) / (np.exp((dv / q)) - 1)
        return f_lexp
    
    
    def f_exp(self, v, R, v12, q):
        dv = -(v - v12)
        f_exp = R * np.exp((dv / q))
        return f_exp
    
    
    def f_sigm(self, v, R, v12, q):
        dv = -(v - v12)
        f_sigm = R / (1 + np.exp((dv / q)))
        return f_sigm
    
    def compute_kinetic_variables(self, vm):
        Rma = self.channel_params.get("Rma_CaHVA", 1)
        Rmb = self.channel_params.get("Rmb_CaHVA", 1)
        v12ma = self.channel_params.get("v12ma_CaHVA", 1)
        v12mb = self.channel_params.get("v12mb_CaHVA", 1)
        qma = self.channel_params.get("qma_CaHVA", 1)
        qmb = self.channel_params.get("qmb_CaHVA", 1)
        Rha = self.channel_params.get("Rha_CaHVA", 1)
        Rhb = self.channel_params.get("Rhb_CaHVA", 1)
        v12ha = self.channel_params.get("v12ha_CaHVA", 1)
        v12hb = self.channel_params.get("v12hb_CaHVA", 1)
        qha = self.channel_params.get("qha_CaHVA", 1)
        qhb = self.channel_params.get("qhb_CaHVA", 1)
        
        alpm = self.f_lexp(vm, Rma, v12ma, qma)
        betm = self.f_exp(vm, Rmb, v12mb, qmb)
        mTau = 1 / (self.tadj * (alpm + betm))
        mInf = alpm / (alpm + betm)
        alph = self.f_exp(vm, Rha, v12ha, qha)
        beth = self.f_sigm(vm, Rhb, v12hb, qhb)
        hTau = 1 / (self.tadj * (alph + beth))
        hInf = alph / (alph + beth)
        return mInf, mTau, hInf, hTau

    def update_states(self, states, dt, v, params):
        m = states['m_CaHVA']
        h = states['h_CaHVA']
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        new_m = exponential_euler(m, dt, mInf, mTau)
        new_h = exponential_euler(h, dt, hInf, hTau)
        return {
            "m_CaHVA": new_m,
            "h_CaHVA": new_h
            }

    def compute_current(self, states, v, params):
        m = states['m_CaHVA']
        h = states['h_CaHVA']
        gbar = params["gbar_CaHVA"]
        E = params.get("E_ca", 140)
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        g = self.tadj * gbar * m**2 * h**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        mInf, mTau, hInf, hTau = self.compute_kinetic_variables(v)
        return {
            "m_CaHVA": mInf,
            "h_CaHVA": hInf
            }

