# This Python channel class was automatically generated from a MOD file
# using DendroTweaks toolbox, dendrotweaks.dendrites.gr
# Manually fixed: self.temperature, formatting, tadj conductance scaling


from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler
import jax.numpy as np

class Ka(Channel):
    """
    K-A channel from Klee Ficker and Heinemann

    NOTE: Unlike most channels, Ka does NOT scale conductance by tadj.
    Temperature dependence is embedded directly in the kinetic rate
    functions via the Boltzmann factor exp(z*F*V / R*T), where T is
    the absolute temperature. The tadj in the PROCEDURE only scales
    the time constant denominator.
    """

    def __init__(self, name="Ka"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
        self.channel_params = {
            "gbar_Ka": 0.0,
            "vhalfn_Ka": 11,
            "vhalfl_Ka": -56,
            "a0l_Ka": 0.05,
            "a0n_Ka": 0.05,
            "zetan_Ka": -1.5,
            "zetal_Ka": 3,
            "gmn_Ka": 0.55,
            "gml_Ka": 1,
            "lmin_Ka": 2,
            "nmin_Ka": 0.1,
            "pw_Ka": -1,
            "tq_Ka": -40,
            "qq_Ka": 5,
            "q10_Ka": 5,
            "qtl_Ka": 1,
            "temp_Ka": 24
            }
        self.channel_states = {
            "n_Ka": 0.0,
            "l_Ka": 0.0
            }
        self._state_powers = {
            "n_Ka": {'power': 1},
            "l_Ka": {'power': 1}
            }
        self.ion = "k"
        self.current_name = "i_k"

        self.independent_var_name = "v"
        self.tadj = 1
        self.temperature = 37  # celsius — used directly in kinetic functions

    def set_tadj(self, temperature):
        """
        Set temperature for kinetic rate functions.

        Ka's conductance is NOT temperature-scaled (no tadj in BREAKPOINT).
        The temperature dependency enters through the Boltzmann factors in
        alpn/betn/alpl/betl which use absolute temperature directly.
        We keep self.tadj for the time constant scaling in compute_kinetic_variables.
        """
        self.temperature = temperature
        q10 = self.channel_params.get("q10_Ka")
        reference_temp = self.channel_params.get("temp_Ka")
        if q10 is not None and reference_temp is not None:
            self.tadj = q10 ** ((temperature - reference_temp) / 10)

    def __getitem__(self, item):
        return self.channel_params[item]

    def __setitem__(self, item, value):
        self.channel_params[item] = value

    def alpn(self, v):
        vhalfn = self.channel_params.get("vhalfn_Ka", 1)
        zetan = self.channel_params.get("zetan_Ka", 1)
        pw = self.channel_params.get("pw_Ka", 1)
        tq = self.channel_params.get("tq_Ka", 1)
        qq = self.channel_params.get("qq_Ka", 1)

        zeta = zetan + (pw / (1 + np.exp(((v - tq) / qq))))
        alpn = np.exp(((((0.001 * zeta) * (v - vhalfn)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return alpn

    def betn(self, v):
        vhalfn = self.channel_params.get("vhalfn_Ka", 1)
        zetan = self.channel_params.get("zetan_Ka", 1)
        gmn = self.channel_params.get("gmn_Ka", 1)
        pw = self.channel_params.get("pw_Ka", 1)
        tq = self.channel_params.get("tq_Ka", 1)
        qq = self.channel_params.get("qq_Ka", 1)

        zeta = zetan + (pw / (1 + np.exp(((v - tq) / qq))))
        betn = np.exp((((((0.001 * zeta) * gmn) * (v - vhalfn)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return betn

    def alpl(self, v):
        vhalfl = self.channel_params.get("vhalfl_Ka", 1)
        zetal = self.channel_params.get("zetal_Ka", 1)

        alpl = np.exp(((((0.001 * zetal) * (v - vhalfl)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return alpl

    def betl(self, v):
        vhalfl = self.channel_params.get("vhalfl_Ka", 1)
        zetal = self.channel_params.get("zetal_Ka", 1)
        gml = self.channel_params.get("gml_Ka", 1)

        betl = np.exp((((((0.001 * zetal) * gml) * (v - vhalfl)) * 96480.0) / (8.315 * (273.16 + self.temperature))))
        return betl

    def compute_kinetic_variables(self, v):
        a0n = self.channel_params.get("a0n_Ka", 1)
        lmin = self.channel_params.get("lmin_Ka", 1)
        nmin = self.channel_params.get("nmin_Ka", 1)
        qtl = self.channel_params.get("qtl_Ka", 1)

        a = self.alpn(v)
        nInf = 1 / (1 + a)
        nTau = self.betn(v) / ((self.tadj * a0n) * (1 + a))
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
        n = states['n_Ka']
        l = states['l_Ka']
        nInf, nTau, lInf, lTau = self.compute_kinetic_variables(v)
        new_n = exponential_euler(n, dt, nInf, nTau)
        new_l = exponential_euler(l, dt, lInf, lTau)
        return {
            "n_Ka": new_n,
            "l_Ka": new_l
            }

    def compute_current(self, states, v, params):
        n = states['n_Ka']
        l = states['l_Ka']
        gbar = params["gbar_Ka"]
        E = params.get("E_k", -80)
        g = gbar * n * l  # No tadj scaling on conductance (matches MOD BREAKPOINT)
        return g * (v - E)

    def init_state(self, states, v, params, delta_t):
        nInf, nTau, lInf, lTau = self.compute_kinetic_variables(v)
        return {
            "n_Ka": nInf,
            "l_Ka": lInf
            }
