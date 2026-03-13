#!/usr/bin/env python

"""script for data of Figure 1F
stimulates neuron somatically to fire at different frequencies"""

# _title_     : critical_frequency.py
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
from config_validation import *
from sim import *



path = './'

NO_REPS = 5 # 5 somatic stimulations
RESET = True
DT=0.1 # ms, set the integration step
POST_AMP = 0.3 # nA, amplitude of current injection to trigger AP/bAP
WARM_UP=1000 # ms
DELTA_T=10 #ms, does not matter in this case, only somatic stimulation
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
            start_t = (0 + delta_t + i* (1000.0/freq) + WARM_UP)
        end_t = (start_t+2)
        if(test) :
            print 'start_t=%g, end_t=%g (t_stop=%g, len(trace)=%f)' % (start_t,end_t,t_stop,len(trace))
        trace[start_t/DT:end_t/DT] = POST_AMP
    return trace

def main():

    my_rawdata = {}

    sim_params = params['sim']
    #inputs
    frequencies = np.arange(10,100,10)

    distal_integral = np.zeros((len(frequencies)))

    for i, freq in enumerate(frequencies):
        cell = Neuron()
        sim = Simulation(cell,sim_params)
        sim.dt = DT
    
    
        # somatic stimulation
        ic = h.IClamp(cell.soma(0.5))
        ic.delay = 0
        ic.dur=1e9
        total_time = WARM_UP+NO_REPS*(1000.0/10)+100
        current_trace = _get_current_trace(freq,DELTA_T,total_time,pre=False)
        current_vec = h.Vector(current_trace)
        current_vec.play(ic._ref_amp,DT)
    
        sim.sim_time = total_time
    
        # recording
        trec = h.Vector()
        trec.record(h._ref_t)
    
        sim.set_highestres_recording()    
        sim.go()
        t = np.array(trec)
        recording = sim.get_highestres_recording()
        num_spikes, rate = sim.get_rate(0,total_time)
        
        distal_voltage = recording[20,:]
        index = int(1000/0.1) # take integral from 1000 ms to end of simulation (600ms interval)
        distal_integral[i] = np.sum(distal_voltage[index:]-(-75))
    
    fig = plt.figure()
    plt.plot(frequencies,distal_integral/distal_integral[-1],'k',marker='o',mfc="w",mew=1)
    plt.xlabel("frequency [Hz]", fontsize = 'large')
    plt.ylabel("Integral of distal voltage", fontsize = 'large')
    plt.axis(xmin = 0,xmax=90)
    plt.axis(ymin = 0.2,ymax=1.1)
    plt.savefig('%s/crit_freq.eps'%(savepath))    
    

    my_rawdata['t'] = t
    my_rawdata['num_spikes'] = num_spikes
    my_rawdata['rate'] = rate
    my_rawdata['recording'] = recording

    rawdata = {'raw_data': my_rawdata}

    return rawdata



if __name__ == '__main__':

    rawdata = main()

