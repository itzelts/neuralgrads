
NEURON {
	SUFFIX kca
	USEION k READ ek WRITE ik
	USEION ca READ cai
	RANGE gbar, g, ik
}

UNITS {
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
	(mM) = (milli/liter)
}

PARAMETER {
	gbar = 0.0 (S/cm2)
	caix = 1
	Ra = 0.01 (/ms)
	Rb = 0.02 (/ms)
	q10 = 2.3
	temp = 23 (degC)
}

ASSIGNED {
	v	(mV)
	ek	(mV)
	cai	(mM)
	ik	(mA/cm2)
	g	(S/cm2)
	celsius (degC)
	nInf
	nTau	(ms)
}

STATE {
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gbar*n
	ik = g*(v-ek)
}

DERIVATIVE states {
	rates(cai)
	n' = (nInf-n)/nTau
}

INITIAL {
	rates(cai)
	n = nInf
}

PROCEDURE rates(cai(mM)) {
	LOCAL a, qt
	qt = q10^((celsius-temp)/10)

	a = Ra*cai^caix
	nTau = 1/(a+Rb)/qt
	nInf = a/(a+Rb)
}
