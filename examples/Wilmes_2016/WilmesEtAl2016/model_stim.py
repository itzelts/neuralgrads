#!/usr/bin/env python

"""produces voltage and current traces for Figure 1B-E"""

# _title_     : model_stim.py
# _author_     : Katharina Anna Wilmes
# _mail_     : katharina.anna.wilmes __at__ cms.hu-berlin.de


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
from config_model_stim import *
from sim import *


path = './'

SCEN = 3
NO_REPS = 1
RESET = True
DT=0.1 # ms, set the integration step
POST_AMP = 0.3# nA, amplitude of current injection to trigger the AP/bAP
WARM_UP=1000 # ms
AP_DELAY = 0 #ms, AP needs 4ms after stimulation to become initiated
identifier = '2015-01-13-00h00m00s'
savepath = '%s%s'%(path,identifier)
if not os.path.exists(savepath):
	os.mkdir(savepath)

def _get_current_trace(freq,delta_t,t_stop,pre=False,test=True) :
    trace = np.zeros(t_stop/DT)
    for i in range(NO_REPS) :
        if(pre) :
            start_t = (0 + i* (1000.0/freq) + WARM_UP)
        else :
            start_t = (0 + delta_t - AP_DELAY + i* (1000.0/freq) + WARM_UP)
        end_t = (start_t+2)
        if(test) :
            print 'start_t=%g, end_t=%g (t_stop=%g, len(trace)=%f)' % (start_t,end_t,t_stop,len(trace))
        trace[start_t/DT:end_t/DT] = POST_AMP
    return trace

def create_syn(syn_type,pos, thresh):
    syn = h.ExpSynSTDP(pos)
    syn.thresh = thresh
    return syn


def main():

    my_rawdata = {}

    #inputs
    freq = params['Input']['freq'].value
    wee = params['Input']['wee'].value
    wee_strong = params['Input']['wee_strong'].value

    # Synapses
    scen = SCEN
    delta_t = params['STDP']['delta_t'].value
    
    sim_params = params['sim']

    source = False

    cell = Neuron()
    sim = Simulation(cell,sim_params)
    sim.dt = DT
    sim.v_init = -70
    total_time = WARM_UP+NO_REPS*(1000.0/freq)+100

    if (scen == 1) or (scen == 3):
        # trigger AP/bAP
        ic = h.IClamp(cell.soma(0.5))
        ic.delay = 0
        ic.dur=1e9
        current_trace = _get_current_trace(freq,delta_t,total_time,pre=False)
        current_vec = h.Vector(current_trace)
        current_vec.play(ic._ref_amp,DT)

    if (scen == 2) or (scen == 3) or (scen==4):
        # distal excitation
        syn = h.Exp2Syn(cell.branches[0](0.1))
        if scen == 4:
            # strong distal excitation
            weight = wee_strong
        else:
            weight = wee
        syn.e = 0
        syn.tau1 = 0.5
        syn.tau2 = 2
        interval = 1000.0/freq
        exstim = h.NetStim()
        exstim.number = NO_REPS
        exstim.interval = interval
        exstim.start = WARM_UP
        exstim.noise= 0
        nc = h.NetCon(exstim,syn,0,0,weight)

    sim.sim_time = total_time

    # recording
    trec = h.Vector()
    trec.record(h._ref_t)
    vrec = h.Vector()
    vrec.record(cell.soma(0.5)._ref_v)
    vdrec = h.Vector()
    vdrec.record(cell.branches[0](0.5)._ref_v)
    vbrec = h.Vector()
    vbrec.record(cell.basal_main(0.5)._ref_v)
    vorec = h.Vector()
    vorec.record(cell.oblique_branch(0.9)._ref_v)
    if (scen == 2) or (scen == 4):
        currentrec = h.Vector()
        currentrec.record(syn._ref_i)

    # record state vars
    vit2m, vit2h, vscam, vscah, vkcan, vna3dendm, vna3dendh = [h.Vector() for x in xrange(7)]
    vit2m.record(cell.branches[0](0.5).it2._ref_m)
    vit2h.record(cell.branches[0](0.5).it2._ref_h)
    vscam.record(cell.branches[0](0.5).sca._ref_m)
    vscah.record(cell.branches[0](0.5).sca._ref_h)
    vkcan.record(cell.branches[0](0.5).kca._ref_n)
    vna3dendm.record(cell.branches[0](0.5).na3dend._ref_m)
    vna3dendh.record(cell.branches[0](0.5).na3dend._ref_h)

    # run simulation
    sim.go()

    t = np.array(trec)
    v = np.array(vrec)
    vd = np.array(vdrec)
    vb = np.array(vbrec)
    vo = np.array(vorec)

    it2m = np.array(vit2m)
    it2h = np.array(vit2h)
    scam = np.array(vscam)
    scah = np.array(vscah)
    kcan = np.array(vkcan)
    na3dendm = np.array(vna3dendm)
    na3dendh = np.array(vna3dendh)

    # plot traces
    if scen == 1:
        step = np.array(current_vec)
        x = np.arange(len(step))
        plt.figure()
        plt.plot(x*DT,step,'k',label='vd')
        plt.xlim((990,1100))
        plt.ylim((0,3.6))
        plt.xlabel("Time [ms]", fontsize = 'large')
        plt.ylabel("Current [nA]", fontsize = 'large')
        plt.savefig('%s/step.eps'%(savepath))

    if (scen == 2) or (scen == 4):
        crec = np.array(currentrec)
        x = np.arange(len(crec))
        plt.figure()
        plt.plot(x*DT,abs(crec),'r',label='vd')
        plt.xlim((990,1100))
        plt.ylim((0,3.6))
        plt.xlabel("Time [ms]", fontsize = 'large')
        plt.ylabel("Current [nA]", fontsize = 'large')
        plt.savefig('%s/crec%d.eps'%(savepath,scen))
    plt.figure()
    plt.plot(t,v,'k',label='v')
    plt.hold(True)
    plt.plot(t,vd,'r',label='vd')
    plt.xlabel("Time [ms]", fontsize = 'large')
    plt.xlim((990,1100))
    plt.ylim((-80,40))
    plt.ylabel("Voltage [mV]", fontsize = 'large')
    plt.savefig('%s/vrec%d.eps'%(savepath,scen))
    
    plt.figure()
    plt.plot(t,it2m,'k-',label='v')
    plt.hold(True)
    plt.plot(t,it2h,'k--',label='v')
    plt.plot(t,scam,'r-',label='v')
    plt.plot(t,scah,'r--',label='v')
    plt.plot(t,kcan,'g-',label='vd')
    plt.plot(t,na3dendm,'m-',label='v')
    plt.plot(t,na3dendh,'m--',label='v')

    plt.xlabel("Time [ms]", fontsize = 'large')
    plt.xlim((990,1100))
    #plt.ylim((-80,40))
    plt.ylabel("state", fontsize = 'large')
    plt.savefig('%s/statevars%d.eps'%(savepath,scen))

    # delete
    del(cell)
    del(sim)

    del(trec);del(vrec)

    my_rawdata['t'] = t
    my_rawdata['v'] = v
    my_rawdata['vd'] = vd
    my_rawdata['vb'] = vb
    my_rawdata['vo'] = vo
    my_rawdata['it2m'] = it2m
    my_rawdata['it2h'] = it2h
    my_rawdata['scam'] = scam
    my_rawdata['scah'] = scah
    my_rawdata['kcan'] = kcan
    my_rawdata['na3dendm'] = na3dendm
    my_rawdata['na3dendh'] = na3dendh


    rawdata = {'raw_data': my_rawdata}

    return rawdata



if __name__ == '__main__':

    rawdata = main()
