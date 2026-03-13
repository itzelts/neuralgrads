#!/usr/bin/env python

"""produces data for table with voltage traces in Figures 3A and 4A,
"""

# _title_     : traces.py
# _author_     : Katharina Anna Wilmes
# _mail_     : katharina.anna.wilmes __at__ cms.hu-berlin.de


# --Imports--
import sys
import os

import math
import matplotlib.pyplot as plt
import numpy as np
from time import time
import neuron
from neuron import h

from neuronmodel import *
from config_traces_ca import *
from sim import *

# set path for saving data
path = './'


NO_REPS = 1
RESET = True
DT=0.1 # ms, set the integration step
POST_AMP = 0.3# nA, amplitude of current injection to trigger AP/bAP
WARM_UP=1000 # ms, until steady state
AP_DELAY = 4 #ms, AP needs 4ms after stimulation to become initiated
identifier = '2015-01-13-00h00m00s' # folder name
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
    
def traces(t, AP, AP_distal, AP_oblique, AP_basal, inh_times, scen, path=None):

    site = [AP, AP_distal, AP_oblique, AP_basal]
    sitename = ['axon','distal','oblique','basal']
    # Plot data
    fig = plt.figure(1)
    plt.clf()
    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
    a = np.ones((len(inh_times))) * -75

    for i in range(1,5):
        ax = plt.subplot(2,2,i, autoscale_on = False)
        p2 = plt.plot(t,site[i-1],'k', lw=1, label = 'scenario')
        plt.hold(True)
        plt.plot(inh_times, a, '|',color = '#FF8300')
        plt.axis(xmin = 990, xmax = 1100, ymin=-80, ymax=50)
        plt.yticks(np.arange(-80,50,10))
        plt.title(sitename[i-1])
    plt.savefig('%s/publish%s.eps'%(path, scen))


def main():

    my_rawdata = {}

    # inputs
    freq = params['Input']['freq'].value
    wee = params['Input']['wee'].value
    wee_strong = params['Input']['wee_strong'].value

    # Scenario
    scen = params['Synapse']['scen'].value
    #shunt_pos = params['Synapse']['shunt_pos'].value
    delta_t = params['STDP']['delta_t'].value

    sim_params = params['sim']

    # inhibition
    shunt_params = params['shunt']
    distal_shunt_weight = params['shunt']['distal_shunt_weight'].value
    proximal_shunt_weight = params['shunt']['proximal_shunt_weight'].value
    basal_shunt_weight = params['shunt']['basal_shunt_weight'].value
    distal_shunt_pos = params['shunt']['distal_shunt_pos'].value
    proximal_shunt_pos = params['shunt']['proximal_shunt_pos'].value
    basal_shunt_pos = params['shunt']['basal_shunt_pos'].value
    distal_shunt_compartment = params['shunt']['distal_shunt_compartment']
    proximal_shunt_compartment = params['shunt']['proximal_shunt_compartment']
    basal_shunt_compartment = params['shunt']['basal_shunt_compartment']
    basal_shunt_delay = params['shunt']['basal_shunt_delay'].value
    freq_i = params['Synapse']['interneuron_frequency'].value
    reps_inh = params['Synapse']['interneuron_reps'].value
    shunt_delay = params['shunt']['shunt_delay'].value
    shunt_pos_range = np.arange(4)

    source = False

    for shunt_pos in shunt_pos_range:
    
        cell = Neuron()
        sim = Simulation(cell,sim_params)
        sim.dt = DT
        sim.v_init = -70
        total_time = WARM_UP+NO_REPS*(1000.0/freq)+100
        interval = 1000.0/freq
    
    
        # somatic stimulation
        ic = h.IClamp(cell.soma(0.5))
        ic.delay = 0
        ic.dur=1e9
        #print 'testing freq F=%g (t_stop=%i)' % (freq,total_time)
        current_trace = _get_current_trace(freq,delta_t,total_time,pre=False)
        current_vec = h.Vector(current_trace)
        current_vec.play(ic._ref_amp,DT)
    
        # additional distal excitation to trigger calcium spike
        if scen == 3:
            syn = h.Exp2Syn(cell.branches[0](0.1))
            weight = wee
            syn.e = 0
            syn.tau1 = 0.5
            syn.tau2 = 2
            exstim = h.NetStim()
            exstim.number = NO_REPS
            exstim.interval = interval
            exstim.start = WARM_UP
            exstim.noise= 0
            nc = h.NetCon(exstim,syn,0,0,weight)
    
    
    
        inh_delay = WARM_UP + delta_t - AP_DELAY + shunt_delay
        basal_inh_delay = WARM_UP + delta_t - AP_DELAY + basal_shunt_delay
        

        # inhibition
        if shunt_pos == 0:
            print "no inhibition"
        elif shunt_pos == 1: # distal
            sim.set_Shunt(shunt_params, distal_shunt_pos, distal_shunt_weight, distal_shunt_compartment,source,inh_delay,NO_REPS, interval)
        elif shunt_pos == 2: # proximal
            sim.set_Shunt(shunt_params, proximal_shunt_pos, proximal_shunt_weight, proximal_shunt_compartment,source,inh_delay,NO_REPS,interval)
        elif shunt_pos == 3: # basal
            for i in np.arange(NO_REPS):
                sim.set_Shunt(shunt_params, basal_shunt_pos, basal_shunt_weight,basal_shunt_compartment,source,basal_inh_delay+i*interval,no_reps=reps_inh,interval=1000.0/freq_i)
        else:
            raise ValueError
    
        # recording
        trec = h.Vector()
        trec.record(h._ref_t)
        vrec = h.Vector()
        vrec.record(cell.soma(0.5)._ref_v)
        varec = h.Vector()
        varec.record(cell.ais(0.5)._ref_v)
        vdrec = h.Vector()
        vdrec.record(cell.branches[0](0.5)._ref_v)
        vbrec = h.Vector()
        vbrec.record(cell.basal_main(0.5)._ref_v)
        vorec = h.Vector()
        vorec.record(cell.oblique_branch(0.9)._ref_v)
        if scen == 3:
            currentrec = h.Vector()
            currentrec.record(syn._ref_i)
    
    
        if not shunt_pos == 0:
            ipsc = neuron.h.Vector()
            ipsc.record(sim.shunt[0]._ref_i)
            sim.set_synaptic_current_rec()
            sim.set_synaptic_cond_rec()
    
        # record state vars
        vit2m, vit2h, vscam, vscah, vkcan, vna3dendm, vna3dendh = [h.Vector() for x in xrange(7)]
        vit2m.record(cell.branches[0](0.5).it2._ref_m)
        vit2h.record(cell.branches[0](0.5).it2._ref_h)
        vscam.record(cell.branches[0](0.5).sca._ref_m)
        vscah.record(cell.branches[0](0.5).sca._ref_h)
        vkcan.record(cell.branches[0](0.5).kca._ref_n)
        vna3dendm.record(cell.branches[0](0.5).na3dend._ref_m)
        vna3dendh.record(cell.branches[0](0.5).na3dend._ref_h)
    
    
        sim.sim_time = total_time
    
        if not shunt_pos == 0:
            sim.set_synaptic_recording(switch=False,all=False)
    
        sim.go()
        t = np.array(trec)
        v = np.array(vrec)
        vd = np.array(vdrec)
        vb = np.array(vbrec)
        vo = np.array(vorec)
        va = np.array(varec)
    
        it2m = np.array(vit2m)
        it2h = np.array(vit2h)
        scam = np.array(vscam)
        scah = np.array(vscah)
        kcan = np.array(vkcan)
        na3dendm = np.array(vna3dendm)
        na3dendh = np.array(vna3dendh)
    
        if not shunt_pos == 0:
            inh_rec = sim.get_synaptic_recording('dendritic_inh')
            ipsc_rec = sim.get_synaptic_current_rec()
            inhcond = sim.get_synaptic_cond_rec()
        else:
            inh_rec = np.zeros(1)
            ipsc_rec = np.zeros(1)
    
        step = np.array(current_vec)
        x = np.arange(len(step))
        plt.figure()
        plt.plot(x*DT,step,'k',label='vd')
        plt.xlim((990,1100))
        plt.ylim((0,3.6))
        plt.xlabel("Time [ms]", fontsize = 'large')
        plt.ylabel("Current [nA]", fontsize = 'large')
        plt.savefig('%s/step.eps'%(savepath))
    
        if (scen == 3) and (shunt_pos == 0):
            crec = np.array(currentrec)
            x = np.arange(len(crec))
            plt.figure()
            plt.plot(x*DT,abs(crec),'r',label='vd')
            plt.xlim((990,1100))
            plt.ylim((0,3.6))
            plt.xlabel("Time [ms]", fontsize = 'large')
            plt.ylabel("Current [nA]", fontsize = 'large')
            plt.savefig('%s/crec_scen%d_shuntpos%d.eps'%(savepath,scen,shunt_pos))
        plt.figure()
        plt.plot(t,v,'k',label='v')
        plt.hold(True)
        plt.plot(t,vd,'r',label='vd')
        plt.xlabel("Time [ms]", fontsize = 'large')
        plt.xlim((990,1100))
        plt.ylim((-80,40))
        plt.ylabel("Voltage [mV]", fontsize = 'large')
        plt.savefig('%s/vrec_scen%d_shuntpos%d.eps'%(savepath,scen,shunt_pos))
    
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
        plt.savefig('%s/statevars_scen%d_shuntpos%d.eps'%(savepath,scen,shunt_pos))
    

        traces(t,va,vd,vo,vb,inh_rec,"shunt_%d"%(shunt_pos),savepath)
            
    my_rawdata['t'] = t
    my_rawdata['v'] = v
    my_rawdata['vd'] = vd
    my_rawdata['vb'] = vb
    my_rawdata['vo'] = vo
    my_rawdata['va'] = va
    my_rawdata['it2m'] = it2m
    my_rawdata['it2h'] = it2h
    my_rawdata['scam'] = scam
    my_rawdata['scah'] = scah
    my_rawdata['kcan'] = kcan
    my_rawdata['na3dendm'] = na3dendm
    my_rawdata['na3dendh'] = na3dendh
    my_rawdata['inh_times'] = inh_rec
    if not shunt_pos == 0:
        my_rawdata['ipsc'] = ipsc_rec
        my_rawdata['inh_cond'] = inhcond



    #my_rawdata['currentv'] = currentv
    rawdata = {'raw_data': my_rawdata}

    return rawdata


if __name__ == '__main__':

    rawdata = main()
