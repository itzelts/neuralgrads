
NEURON {
	SUFFIX it2
	USEION ca READ eca WRITE ica
	RANGE gbar, g, ica
}

UNITS {
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER {
	gbar = 0.0 (S/cm2)
	vshift = 0 (mV)
	v12m = 50 (mV)
	v12h = 78 (mV)
	vwm = 7.4 (mV)
	vwh = 5.0 (mV)
	am = 3 (mV)
	ah = 85 (mV)
	vm1 = 25 (mV)
	vm2 = 100 (mV)
	vh1 = 46 (mV)
	vh2 = 405 (mV)
	wm1 = 20 (mV)
	wm2 = 15 (mV)
	wh1 = 4 (mV)
	wh2 = 50 (mV)
}

ASSIGNED {
	v	(mV)
	eca	(mV)
	ica	(mA/cm2)
	g	(S/cm2)
	mInf
	mTau	(ms)
	hInf
	hTau	(ms)
}

STATE {
	m
	h
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gbar*m*m*h
	ica = g*(v-eca)
}

DERIVATIVE states {
	rates(v+vshift)
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL {
	rates(v+vshift)
	m = mInf
	h = hInf
}

PROCEDURE rates(v(mV)) {
	mInf = 1.0/(1 + exp(-(v+v12m)/vwm))
	hInf = 1.0/(1 + exp((v+v12h)/vwh))
	mTau = am + 1.0/(exp((v+vm1)/wm1) + exp(-(v+vm2)/wm2))
	hTau = ah + 1.0/(exp((v+vh1)/wh1) + exp(-(v+vh2)/wh2))
}
