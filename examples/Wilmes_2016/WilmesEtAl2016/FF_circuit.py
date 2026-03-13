#!/usr/bin/env python

"""script for feedforward circuit from Figure 7"""

# _title_ 	: FF_circuit.py
# _author_ 	: Katharina Anna Wilmes
# _mail_ 	: katharina.anna.wilmes __at__ cms.hu-berlin.de

# --Imports--
import sys
import os

import math
from time import time
import matplotlib.pyplot as plt
import numpy as np
import neuron
from neuron import h

from neuronmodel import *
from config_circuit import *
from sim import *

#set path for saving data
path = './'

NO_EPSPs = 8
NO_REPS = 1
DT=0.1 # ms, set the integration step
POST_AMP = 0.3 # nA, amplitude of current injection to trigger the AP/bAP
WARM_UP=1000 # ms, until steady state
identifier = '2015-01-13-00h00m00s'
savepath = '%s%s'%(path,identifier)
if not os.path.exists(savepath):
	os.mkdir(savepath)

def get_interneuron(inh_type=0):
    interneuron = neuron.h.Section()
    interneuron.L = 67
    interneuron.diam = 67
    interneuron.nseg = 1
    interneuron.Ra = 100
    interneuron.cm = 1

    interneuron.insert('pas')
    for seg in interneuron:
        seg.pas.g = 0.00015
        seg.pas.e = -70
    # fast interneuron
    if inh_type == 0:
        interneuron.insert('hh2')
        interneuron.vtraub_hh2 = -55 #resting Vm, BJ was -55
        interneuron.gnabar_hh2 = 0.05 #McCormick=15 muS, thal was 0.09
        interneuron.gkbar_hh2 = 0.01 #spike duration of interneurons
        interneuron.ena = 50
        interneuron.ek = -100
    # slower interneuron
    elif inh_type == 1:
        interneuron.insert('hh3')
        interneuron.vtraub_hh3 = -55 #resting Vm, BJ was -55
        interneuron.gnabar_hh3 = 0.05 #McCormick=15 muS, thal was 0.09
        interneuron.gkbar_hh3 = 0.01 #spike duration of interneurons
        interneuron.ena = 50
        interneuron.ek = -100
    else:
        raise ValueError

    return interneuron


def main():
    
    my_rawdata = {}

    #inputs
    freq = params['Input']['freq'].value

    # Synapses
    pos = params['Synapse']['pos'].value

    w_ei = params['Input']['w_ei'].value
    w_ii = params['Input']['w_ii'].value
    w_ee = params['Input']['w_ee'].value

    EPSP_pos = params['Synapse']['EPSP_pos'].value

    inh_delay = params['shunt']['shunt_delay'].value
    inh_pos = params['shunt']['inh_pos'].value
    inh_type = params['shunt']['inh_type'].value
    interneuron_target = params['shunt']['interneuron_target'].value

    sim_params = params['sim']

    for w_ie in [0.0,0.05]:
    
        cell = Neuron()
        sim = Simulation(cell,sim_params)
        sim.dt = DT
        sim.v_init = -70
    
    
        interneuron = get_interneuron(inh_type)
    
    
        # excitatory input neuron
        ns = h.NetStim()
        ns.number = NO_REPS # (average) number of spikes
        ns.interval = 1000.0/freq
        ns.start = WARM_UP # ms (most likely) start time of first spike
    
        # excitatory to inhibitory synapse
        ex_inh = neuron.h.Exp2Syn(interneuron(0.5))
        ex_inh.tau1 = 0.5 # ms rise time
        ex_inh.tau2 = 2 # ms decay time
        ex_inh.e = 0 # mV reversal p
    
        # excite interneuron
        nc_ex_inh = h.NetCon(ns,ex_inh,1,0,w_ei)
    
        # VIP interneuron
        ns_VIP = h.NetStim()
        ns_VIP.number = NO_REPS # (average) number of spikes
        ns_VIP.interval = 1000.0/freq
        ns_VIP.start = WARM_UP # ms (most likely) start time of first spike
    
        #inhibitory to inhibitory synapse
        inh_inh = neuron.h.Exp2Syn(interneuron(0.5))
        inh_inh.tau1 = 0.5 # ms rise time
        inh_inh.tau2 = 5 # ms decay time
        inh_inh.e = -73 # mV reversal p
        w_ii = float(params['Input']['w_ii'].value)
    
        nc_inh_inh = h.NetCon(ns_VIP,inh_inh,1,0,w_ii)
    
    
        # excitatory to excitatory synapse
        ex_ex = h.List()
        nc_ex = h.List()
        branch = 0
        for i in range(1,NO_EPSPs+1):
            pos = (i) * 0.1
            ex = neuron.h.Exp2Syn(cell.apical(pos))
            w = w_ee/NO_EPSPs #0.016
            ex.tau1 = 0.5 # ms rise time
            ex.tau2 = 2 # ms decay time
            ex.e = 0 # mV reversal potential
            # excite postsynaptic cell
            nc = h.NetCon(ns,ex,1,0,w)
            ex_ex.append(ex)
            nc_ex.append(nc)
    
        #inhibitory to excitatory synapse
        if interneuron_target == 0:
            inh_ex = neuron.h.Exp2Syn(cell.apical_prox(inh_pos))
        elif interneuron_target == 1:
            inh_ex = neuron.h.Exp2Syn(cell.apical(inh_pos))
    
        inh_ex.tau1 = 0.5 # ms rise time
        inh_ex.tau2 = 5 # ms decay time
        inh_ex.e = -73 # mV reversal p
    
        # connect interneuron to postsynaptic cell
        nc_inh_ex = h.NetCon(interneuron(0.5)._ref_v, inh_ex,1,inh_delay,w_ie, sec = interneuron)
    
        nc_vec = h.Vector()
        nc_inh_ex.record(nc_vec)
    
        total_time = WARM_UP+NO_REPS*(1000.0/freq)+100
        sim.sim_time = total_time
    
        sim.interneuron = interneuron
    
        # recording
        trec = h.Vector()
        trec.record(h._ref_t)
        vrec = h.Vector()
        vrec.record(cell.soma(0.5)._ref_v)
        vdrec = h.Vector()
        vdrec.record(cell.branches[0](0.5)._ref_v)
        vorec = h.Vector()
        vorec.record(cell.oblique_branch(0.9)._ref_v)
        vbrec = h.Vector()
        vbrec.record(cell.basal_main(0.9)._ref_v)
        vprox = h.Vector()
        vprox.record(cell.apical_prox(0.9)._ref_v)
        vinhrec = h.Vector()
        vinhrec.record(interneuron(0.5)._ref_v)
    
        cadrec = h.Vector()
        cadrec.record(cell.branches[0](0.5)._ref_ica)
    
        # run simulation
        sim.go()
    
        t = np.array(trec)
        v = np.array(vrec)
        vd = np.array(vdrec)
        vo = np.array(vorec)
        vb = np.array(vbrec)
        vinh = np.array(vinhrec)
        vprox = np.array(vprox)
        nc_v = np.array(nc_vec)
        cad = np.array(cadrec)
    
        plt.figure()
        plt.plot(t,v,'r',label='v')
        plt.hold(True)
        plt.plot(t,vo,'g',label='vo')
        if w_ie == 0.05:
            plt.plot(nc_v[0]+inh_delay, -75, '|',color = '#FF8300')
        plt.xlabel("Time [ms]", fontsize = 'large')
        plt.ylabel("Voltage [mV]", fontsize = 'large')
        plt.xlim(1000,1010)
        plt.legend(('soma','dendrite'))
        plt.savefig('%s/vrec_wee%d_type%s_target%s.eps'%(savepath, w_ie*1000,inh_type,interneuron_target))



    my_rawdata['t'] = t
    my_rawdata['v'] = v
    my_rawdata['vd'] = vd
    my_rawdata['vo'] = vo
    my_rawdata['vb'] = vb
    my_rawdata['vinh'] = vinh

    my_rawdata['cad'] = cad

    rawdata = {'raw_data': my_rawdata}


    return rawdata


if __name__ == '__main__':

    rawdata = main()
