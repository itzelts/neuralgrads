#!/usr/bin/env python

"""
script for data from Figure 2A
somatic stimulation, proximal inhibition at 90um from the soma
with varying strength
"""

# _title_     : allornone.py
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
from config_allornone import *
from sim import *


#set path for saving data
path = './'

NO_REPS = 1
DT=0.1 # ms, set the integration step
POST_AMP = 0.3 # nA, amplitude of current injection to trigger the bAP
WARM_UP=1000 # ms, wait until steady state
DELTA_T=0 #ms, not applicable, only somatic stimulation
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
            start_t = (0 + delta_t + i* (1000.0/freq) + WARM_UP)
        end_t = (start_t+2)
        if(test) :
            print 'start_t=%g, end_t=%g (t_stop=%g, len(trace)=%f)' % (start_t,end_t,t_stop,len(trace))
        trace[start_t/DT:end_t/DT] = POST_AMP
    return trace


def main():

    my_rawdata = {}

    # simulation
    sim_params = params['sim']
    
    #inputs
    freq = params['Input']['freq'].value

    # Synapses
    timing_sigma  = params['Synapse']['timing_sigma'].value
    interval = 1000.0/freq
    shunt_params = params['shunt']
    shunt_delay = params['shunt']['shunt_delay'].value
    shunt_number = params['shunt']['shunt_number'].value
    shunt_sigma = params['shunt']['shunt_sigma'].value
    pattern = params['shunt']['shunt_pattern']
    shunt_pos = params['shunt']['exact_shunt_pos'].value
    shunt_compartment = params['shunt']['shunt_compartment']
    DISTRIBUTED = params['shunt']['distributed'].value # distribute in space

    rec_pos = ('soma','30','60','90','130','160','190','220','250','280','310','340','370')    
    shunt_weights = [0, 0.015, 0.05] # no shunt, weak shunt, strong shunt
    # shunt_weights were between 0 and 0.1 in 0.005 steps in Fig 2


    source = False
    peaks = np.zeros((len(shunt_weights), len(rec_pos)))


    for i, shunt_weight in enumerate(shunt_weights):

        cell = Neuron()
        sim = Simulation(cell,sim_params)
        sim.dt = DT
        sim.v_init = -70
    
        #somatic stimulation
        ic = h.IClamp(cell.soma(0.5))
        ic.delay = 0
        ic.dur=1e9
        total_time = WARM_UP+NO_REPS*(1000.0/freq)+100
        current_trace = _get_current_trace(freq,DELTA_T,total_time,pre=False)
        current_vec = h.Vector(current_trace)
        current_vec.play(ic._ref_amp,DT)
    
        sim.sim_time = total_time
        
        if shunt_number == 0:
            shunt_number = shunt_weight/0.001 # divide weight by strength of 1 synapse - 1nS (0.001uS)
        inh_delay = WARM_UP + DELTA_T + shunt_delay
        print shunt_weight
        print i
        if DISTRIBUTED == 1:
            print "distributed"
            pos, timing = sim.set_distributed_shunt(shunt_params, shunt_pos,
            shunt_weight, shunt_compartment, shunt_number, shunt_sigma,
            pattern, timing_sigma, inh_delay, NO_REPS, interval)
        else:
            sim.set_Shunt(shunt_params, shunt_pos, shunt_weight,
            shunt_compartment,source,inh_delay,NO_REPS,interval)

        
        # recording
        trec = h.Vector()
        trec.record(h._ref_t)
        vrec = h.Vector()
        vrec.record(cell.soma(0.5)._ref_v)
        vaisrec = h.Vector()
        vaisrec.record(cell.ais(0.5)._ref_v)
        vorec = h.Vector()
        vorec.record(cell.oblique_branch(0.5)._ref_v)
    
        rec_v = neuron.h.Vector()
        rec_v1 = neuron.h.Vector()
        rec_v2 = neuron.h.Vector()
        rec_v3 = neuron.h.Vector()
        rec_v4 = neuron.h.Vector()
        rec_v5 = neuron.h.Vector()
        rec_v6 = neuron.h.Vector()
        rec_v7 = neuron.h.Vector()
        rec_v8 = neuron.h.Vector()
        rec_v9 = neuron.h.Vector()
        rec_v10 = neuron.h.Vector()
        rec_v.record(cell.soma(0.5)._ref_v)
        rec_v1.record(cell.apical_prox(0.3)._ref_v)
        rec_v2.record(cell.apical_prox(0.6)._ref_v)
        rec_v3.record(cell.apical_prox(0.9)._ref_v)
        rec_v4.record(cell.apical(0.05)._ref_v)
        rec_v5.record(cell.apical(0.2)._ref_v)
        rec_v6.record(cell.apical(0.35)._ref_v)
        rec_v7.record(cell.apical(0.5)._ref_v)
        rec_v8.record(cell.apical(0.65)._ref_v)
        rec_v9.record(cell.apical(0.8)._ref_v)
        rec_v10.record(cell.apical(0.95)._ref_v)
    
        rec_o = h.List()
        for pos in np.arange(0.1,0.9,0.1):
            rec_vo = neuron.h.Vector()
            rec_vo.record(cell.oblique_branch(pos)._ref_v)
            rec_o.append(rec_vo)
    
        rec_vo1 = neuron.h.Vector()
        rec_vo2 = neuron.h.Vector()
        rec_vo3 = neuron.h.Vector()
        rec_vo4 = neuron.h.Vector()
        rec_vo5 = neuron.h.Vector()
        rec_vo6 = neuron.h.Vector()
        rec_vo7 = neuron.h.Vector()
        rec_vo8 = neuron.h.Vector()
        rec_vo9 = neuron.h.Vector()
        rec_vo1.record(cell.oblique_branch(0.1)._ref_v)
        rec_vo2.record(cell.oblique_branch(0.2)._ref_v)
        rec_vo3.record(cell.oblique_branch(0.3)._ref_v)
        rec_vo4.record(cell.oblique_branch(0.4)._ref_v)
        rec_vo5.record(cell.oblique_branch(0.5)._ref_v)
        rec_vo6.record(cell.oblique_branch(0.6)._ref_v)
        rec_vo7.record(cell.oblique_branch(0.7)._ref_v)
        rec_vo8.record(cell.oblique_branch(0.8)._ref_v)
        rec_vo9.record(cell.oblique_branch(0.9)._ref_v)
    
    
        rec_ina_o = h.List()
        rec_ina = neuron.h.Vector()
        rec_ina.record(cell.soma(0.5)._ref_ina)
        rec_ina_o.append(rec_ina)
    
        for pos in np.arange(0.3,1.2,0.3):
            rec_ina = neuron.h.Vector()
            rec_ina.record(cell.apical_prox(pos)._ref_ina)
            rec_ina_o.append(rec_ina)
    
        for pos in np.arange(0.1,1.0,0.1):
            rec_ina = neuron.h.Vector()
            rec_ina.record(cell.oblique_branch(pos)._ref_ina)
            rec_ina_o.append(rec_ina)
    
        rec_ik_o = h.List()
        rec_ik = neuron.h.Vector()
        rec_ik.record(cell.soma(0.5)._ref_ik)
        rec_ik_o.append(rec_ik)
    
        for pos in np.arange(0.3,1.2,0.3):
            rec_ik = neuron.h.Vector()
            rec_ik.record(cell.apical_prox(pos)._ref_ik)
            rec_ik_o.append(rec_ik)
    
        for pos in np.arange(0.1,1.0,0.1):
            rec_ik = neuron.h.Vector()
            rec_ik.record(cell.oblique_branch(pos)._ref_ik)
            rec_ik_o.append(rec_ik)
    
        # run simulation
        sim.go()
    
        # recording to numpy array
        t = np.array(trec)
        v = np.array(vrec)
        vais = np.array(vaisrec)
        vo = np.array(vorec)
    
        recording = np.array((rec_v,rec_v1,rec_v2,rec_v3,
                rec_vo1, rec_vo2,rec_vo3, rec_vo4,rec_vo5,rec_vo6,
                rec_vo7,rec_vo8,rec_vo9))
        recording_apical = np.array((rec_v,rec_v1,rec_v2,
                rec_v3,rec_v4,rec_v5,rec_v6,rec_v7,rec_v8,rec_v9,
                rec_v10))
        rec_ina_o = np.array(rec_ina_o)
        rec_ik_o = np.array(rec_ik_o)
        
        peaks[i] = np.max((recording),1)-(-70)


    # plot    
    norm = peaks[0,:]
    soma_norm = peaks[0,0]
    norm_peaks = peaks/norm
    soma_norm_peaks = peaks/soma_norm
    #effect = (norm[-1]-peaks[:,-1])/norm[-1]

    x = np.arange(np.shape(recording)[0])
    fig = plt.figure(figsize=(13,6))
    ax1  = fig.add_subplot(111)
    ax1.set_position((0.145, 0.15, 0.8, 0.775))
    plots = []
    pl1 = ax1.plot(x,soma_norm_peaks[0,:],color = 'k')
    ax1.hold(True)
    pl2 = ax1.plot(x,soma_norm_peaks[1,:],color = 'g')
    pl3 = ax1.plot(x,soma_norm_peaks[2,:],color = 'r')
    plt.legend(['no shunt','weak shunt','strong shunt'])
    plt.yticks(fontsize = 20)
    plt.xticks(np.arange(len(rec_pos)),rec_pos, fontsize = 20)
    plt.xlabel('dendritic location $[\mu m]$',fontsize = 28) #unit $[\mu m]$
    plt.ylabel('bAP amplitude',fontsize = 28)
    plt.savefig('%s/allornone.eps'%(savepath))



    # data
    my_rawdata['t'] = t
    my_rawdata['v'] = v
    my_rawdata['vo'] = vo
    my_rawdata['vais'] = vais
    my_rawdata['recording'] = recording
    my_rawdata['recording_apical'] = recording_apical
    my_rawdata['rec_ina'] = rec_ina_o
    my_rawdata['rec_ik'] = rec_ik_o

    if DISTRIBUTED:
        my_rawdata['pos'] = pos
        my_rawdata['timing'] = timing


    rawdata = {'raw_data': my_rawdata}

    return rawdata


if __name__ == '__main__':

    rawdata = main()
