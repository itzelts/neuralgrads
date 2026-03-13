#!/usr/bin/env python

"""plasticity pairing protocol used in Figure 6, S2 and S5"""

# _title_     : pairingprotocol.py
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
from config_pairing import *
from sim import *


path = './'


NO_REPS = 5 # 100 pairings
RESET = True
DT=0.1 # ms, set the integration step
POST_AMP = 0.3 # nA, amplitude of current injection to trigger the POST-synaptic spike
WARM_UP=1000 # ms
AP_DELAY = 4 #ms, AP needs 4ms after stimulation to become initiated
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

def create_syn(syn_type,pos, thresh, w_max, potentiation_factor, depression_factor):

    syn = h.ExpSynSTDP(pos)
    syn.thresh = thresh
    syn.wmax = w_max
    syn.dd == depression_factor
    syn.dp = potentiation_factor
    return syn


def main():

    my_rawdata = {}

    #inputs
    freq = params['Input']['freq'].value

    # Synapses
    pos = params['set_me']['pos'].value
    shunt_pos = params['set_me']['shunt_pos'].value
        
    syn_type = params['Synapse']['syn_type']
    freq_i = params['Synapse']['interneuron_frequency'].value
    reps_inh = params['Synapse']['interneuron_reps'].value
    total_distal_weight = params['Synapse']['total_distal_weight'].value
    
    proximal_shunt_pos = params['shunt']['proximal_shunt_pos'].value
    distal_shunt_pos = params['shunt']['distal_shunt_pos'].value
    basal_shunt_pos = params['shunt']['basal_shunt_pos'].value
    proximal_shunt_compartment = params['shunt']['proximal_shunt_compartment']
    distal_shunt_compartment = params['shunt']['distal_shunt_compartment']
    basal_shunt_compartment = params['shunt']['basal_shunt_compartment']
    shunt_delay = params['shunt']['shunt_delay'].value
    distal_shunt_weight = params['shunt']['distal_shunt_weight'].value
    proximal_shunt_weight = params['shunt']['proximal_shunt_weight'].value
    basal_shunt_weight = params['shunt']['basal_shunt_weight'].value
    basal_shunt_delay = params['shunt']['basal_shunt_delay'].value

    thresh = params['STDP']['thresh'].value
    ca_thresh = params['STDP']['ca_thresh'].value
    potentiation_factor = params['STDP']['potentiation_factor'].value
    depression_factor = params['STDP']['depression_factor'].value
    w_max = params['STDP']['w_max'].value

    sim_params = params['sim']
    shunt_params = params['shunt']   

    source = False

    delta_t_range = np.arange(-20,21,1)

    weight_change = np.zeros((len(delta_t_range)))


    for i, delta_t in enumerate(delta_t_range):
        print "iteration %d from %d"%(i+1,len(delta_t_range)) 

        cell = Neuron()
        sim = Simulation(cell,sim_params)
        sim.dt = DT
        sim.v_init = -70
    
        # set plastic synapses
        if pos == 1:
            syn = create_syn(syn_type,cell.basal_main(0.5),thresh, w_max, potentiation_factor, depression_factor)
            weight = params['Synapse']['basal_weight'].value
        elif pos == 2:
            syn = create_syn(syn_type,cell.oblique_branch(0.5),thresh, w_max, potentiation_factor, depression_factor)
            weight = params['Synapse']['oblique_weight'].value
        elif pos == 3:
            #syn = create_syn(syn_type,cell.branches[0](0.1),thresh)
            syn = h.ExpSynCaSTDP(cell.branches[0](0.1))
            weight = params['Synapse']['distal_weight'].value
        else:
            raise ValueError
    
    
        exnc = h.List()
        interval = 1000.0/freq
        exstim = h.NetStim()
        exstim.number = NO_REPS
        exstim.interval = interval
        exstim.start = WARM_UP
        exstim.noise= 0
    
        exnc.append(h.NetCon(exstim,syn,0,0,weight))
        if pos == 3:
            # distal excitation to trigger calcium spike
            syn1 = h.Exp2Syn(cell.branches[0](0.1))
            syn1.e = 0
            syn1.tau1 = 0.5
            syn1.tau2 = 2
            weight1 = total_distal_weight-weight
            exnc.append(h.NetCon(exstim,syn1,0,0,weight1))
    
        # somatic current injection to trigger postsynaptic spike
        ic = h.IClamp(cell.soma(0.5))
        ic.delay = 0
        ic.dur=1e9
        total_time = WARM_UP+NO_REPS*(1000.0/freq)+100
        current_trace = _get_current_trace(freq,delta_t,total_time,pre=False)
        current_vec = h.Vector(current_trace)
        current_vec.play(ic._ref_amp,DT)
    
        sim.sim_time = total_time
    
        inh_delay = WARM_UP + delta_t - AP_DELAY + shunt_delay
        basal_inh_delay = WARM_UP + delta_t - AP_DELAY + basal_shunt_delay
    
        # inhibition
        if shunt_pos == 0:
            print "no inhibition"
        elif shunt_pos == 1: # distal
            sim.set_Shunt(shunt_params, distal_shunt_pos, shunt_weight, distal_shunt_compartment, source, inh_delay, NO_REPS, interval)
        elif shunt_pos == 2: # proximal
            sim.set_Shunt(shunt_params, proximal_shunt_pos, shunt_weight, proximal_shunt_compartment, source, inh_delay, NO_REPS, interval)
        elif shunt_pos == 3: # basal
            for i in np.arange(NO_REPS):
                sim.set_Shunt(shunt_params, basal_shunt_pos, shunt_weight, basal_shunt_compartment, source, basal_inh_delay+i*interval, no_reps = reps_inh, interval=1000.0/freq_i)                
        else:
            raise ValueError
    
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
    
        # record state vars
        vit2m, vit2h, vscam, vscah, vkcan, vna3dendm, vna3dendh = [h.Vector() for x in xrange(7)]
        vit2m.record(cell.branches[0](0.5).it2._ref_m)
        vit2h.record(cell.branches[0](0.5).it2._ref_h)
        vscam.record(cell.branches[0](0.5).sca._ref_m)
        vscah.record(cell.branches[0](0.5).sca._ref_h)
        vkcan.record(cell.branches[0](0.5).kca._ref_n)
        vna3dendm.record(cell.branches[0](0.5).na3dend._ref_m)
        vna3dendh.record(cell.branches[0](0.5).na3dend._ref_h)
    
        if syn_type == 'additive':
            if True:
                grec = h.Vector()
                grec.record(exnc[0]._ref_weight[1])
                wrec = h.Vector()
                wrec.record(exnc[0]._ref_weight[3])
            else:
                wrec = h.Vector()
                wrec.record(syn._ref_w)
    
        if pos == 3:
            carec = h.Vector()
            carec.record(cell.branches[0](0.1)._ref_cai)
    
    
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
    
    
        if syn_type == 'additive':
            w = np.array(wrec)
        if pos ==3:
            car = np.array(carec)


        weight_change[i] = (w[-1]-w[0])

    norm_factor = np.max(weight_change)
    print np.shape(norm_factor)

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(delta_t_range,weight_change[:]/norm_factor)
    ax.spines['left'].set_position('zero')
    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    plt.ylim(-1,1.1)
    plt.xlim(-20,20)
    plt.ylabel('$\Delta$ w')
    plt.xlabel('$\Delta$ t')
    plt.savefig('%s/weight_change.eps'%(savepath))

    # delete
    del(cell)
    del(sim)
    del(syn)
    del(exstim)
    del(exnc)
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


    if pos ==3:
        my_rawdata['ca'] = car

    if syn_type == 'additive':
        my_rawdata['w'] = w

    rawdata = {'raw_data': my_rawdata}

    return rawdata

if __name__ == '__main__':

    rawdata = main()
