# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class na3shifted(Channel):
    """
    na3
    """

    def __init__(self, name="na3shifted"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_na3shifted": 0.01,
            "vshift_na3shifted": 10,
            "tha_na3shifted": -30,
            "qa_na3shifted": 7.2,
            "Ra_na3shifted": 0.4,
            "Rb_na3shifted": 0.124,
            "thi1_na3shifted": -45,
            "thi2_na3shifted": -45,
            "qd_na3shifted": 1.5,
            "qg_na3shifted": 1.5,
            "mmin_na3shifted": 0.02,
            "hmin_na3shifted": 0.5,
            "q10_na3shifted": 2,
            "Rg_na3shifted": 0.01,
            "Rd_na3shifted": 0.03,
            "qq_na3shifted": 10,
            "tq_na3shifted": -55,
            "thinf_na3shifted": -50,
            "qinf_na3shifted": 4,
            "vhalfs_na3shifted": -60,
            "a0s_na3shifted": 0.0003,
            "zetas_na3shifted": 12,
            "gms_na3shifted": 0.2,
            "smax_na3shifted": 10,
            "vvh_na3shifted": -58,
            "vvs_na3shifted": 2,
            "ar2_na3shifted": 1
            }
        self.channel_states = {
            "m_na3shifted": 0.0,
            "h_na3shifted": 0.0,
            "s_na3shifted": 0.0
            }
        self._state_powers = {
            "m_na3shifted": {'power': 3},
            "h_na3shifted": {'power': 1},
            "s_na3shifted": {'power': 1}
            }
        self.ion = "na"
        self.current_name = "i_na"

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
        q10 = self.channel_params.get(f"q10_na3shifted")
        reference_temp = self.channel_params.get(f"temp_na3shifted")
        if q10 is None or reference_temp is None:
            self.tadj = 1
            print(f"Warning: q10 or reference temperature not set for {self.name}. Using default tadj = 1.")
        else:
            self.tadj = q10 ** ((temperature - reference_temp) / 10)

    def __getitem__(self, item):
        return self.channel_params[item]

    def __setitem__(self, item, value):
        self.channel_params[item] = value
        
    
    def alpv(self, v):
        vvh = self.channel_params.get("vvh_na3shifted", 1)
        vvs = self.channel_params.get("vvs_na3shifted", 1)
        
        alpv = 1 / (1 + np.exp(((v - vvh) / vvs)))
        return alpv
    
    
    def alps(self, v):
        vhalfs = self.channel_params.get("vhalfs_na3shifted", 1)
        zetas = self.channel_params.get("zetas_na3shifted", 1)
        
        alps = np.exp(((((0.001 * zetas) * (v - vhalfs)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return alps
    
    
    def bets(self, v):
        vhalfs = self.channel_params.get("vhalfs_na3shifted", 1)
        zetas = self.channel_params.get("zetas_na3shifted", 1)
        gms = self.channel_params.get("gms_na3shifted", 1)
        
        bets = np.exp((((((0.001 * zetas) * gms) * (v - vhalfs)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return bets
    
    
    def trap0(self, v, th, a, q):
        conditions = [np.abs((v - th)) > 1e-06, ~(np.abs((v - th)) > 1e-06)]
        choices = [(a * (v - th)) / (1 - np.exp((-(v - th) / q))), a * q]
        trap0 = np.select(conditions, choices)
        return trap0
    
    def compute_kinetic_variables(self, v):
        v = v + self.channel_params.get("vshift_na3shifted", 0)
        tha = self.channel_params.get("tha_na3shifted", 1)
        qa = self.channel_params.get("qa_na3shifted", 1)
        Ra = self.channel_params.get("Ra_na3shifted", 1)
        Rb = self.channel_params.get("Rb_na3shifted", 1)
        thi1 = self.channel_params.get("thi1_na3shifted", 1)
        thi2 = self.channel_params.get("thi2_na3shifted", 1)
        qd = self.channel_params.get("qd_na3shifted", 1)
        qg = self.channel_params.get("qg_na3shifted", 1)
        mmin = self.channel_params.get("mmin_na3shifted", 1)
        hmin = self.channel_params.get("hmin_na3shifted", 1)
        q10 = self.channel_params.get("q10_na3shifted", 1)
        Rg = self.channel_params.get("Rg_na3shifted", 1)
        Rd = self.channel_params.get("Rd_na3shifted", 1)
        thinf = self.channel_params.get("thinf_na3shifted", 1)
        qinf = self.channel_params.get("qinf_na3shifted", 1)
        a0s = self.channel_params.get("a0s_na3shifted", 1)
        smax = self.channel_params.get("smax_na3shifted", 1)
        ar2 = self.channel_params.get("ar2_na3shifted", 1)
        
        qt = q10 ** ((self.temperature - 24) / 10)
        a = self.trap0(v, tha, Ra, qa)
        b = self.trap0(-v, -tha, Rb, qa)
        mTau = (1 / (a + b)) / qt
        conditions = [mTau < mmin, ~(mTau < mmin)]
        choices = [mmin, mTau]
        mTau = np.select(conditions, choices)
        mInf = a / (a + b)
        a = self.trap0(v, thi1, Rd, qd)
        b = self.trap0(-v, -thi2, Rg, qg)
        hTau = (1 / (a + b)) / qt
        conditions = [hTau < hmin, ~(hTau < hmin)]
        choices = [hmin, hTau]
        hTau = np.select(conditions, choices)
        hInf = 1 / (1 + np.exp(((v - thinf) / qinf)))
        c = self.alpv(v)
        sInf = c + (ar2 * (1 - c))
        sTau = self.bets(v) / (a0s * (1 + self.alps(v)))
        conditions = [sTau < smax, ~(sTau < smax)]
        choices = [smax, sTau]
        sTau = np.select(conditions, choices)
        return mInf, mTau, hInf, hTau, sInf, sTau

    def update_states(self, states, dt, v, params):
        m = states['m_na3shifted']
        h = states['h_na3shifted']
        s = states['s_na3shifted']
        mInf, mTau, hInf, hTau, sInf, sTau = self.compute_kinetic_variables(v)
        new_m = exponential_euler(m, dt, mInf, mTau)
        new_h = exponential_euler(h, dt, hInf, hTau)
        new_s = exponential_euler(s, dt, sInf, sTau)
        return {
            "m_na3shifted": new_m,
            "h_na3shifted": new_h,
            "s_na3shifted": new_s
            }

    def compute_current(self, states, v, params):
        m = states['m_na3shifted']
        h = states['h_na3shifted']
        s = states['s_na3shifted']
        gbar = params["gbar_na3shifted"]
        E = params.get("E_na", 60)
        mInf, mTau, hInf, hTau, sInf, sTau = self.compute_kinetic_variables(v)
        g = self.tadj * gbar * m**3 * h**1 * s**1 
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        mInf, mTau, hInf, hTau, sInf, sTau = self.compute_kinetic_variables(v)
        return {
            "m_na3shifted": mInf,
            "h_na3shifted": hInf,
            "s_na3shifted": sInf
            }

