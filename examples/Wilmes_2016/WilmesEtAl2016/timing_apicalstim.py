#!/usr/bin/env python

"""script for producing data for Figures 3C, 4C,
timing and strength of inhibition is varied,
inhibition can be on the basal, proximal or distal apical
dendrite"""

# _title_     : timing_apicalstim.py
# _author_     : Katharina Anna Wilmes
# _mail_     : katharina.anna.wilmes __at__ cms.hu-berlin.de


# --Imports--
import sys
import os
import math
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import neuron
from neuron import h

from neuronmodel import *
from config_timing_apical import *
from sim import *

#set path for saving data
path = './'


NO_EPSPs = 8
NO_REPS = 1
DT=0.1 # ms, set the integration step.
POST_AMP = 0.3 # nA, amplitude of current injection to trigger AP/bAP
WARM_UP=1000 # ms, wait until steady state
AP_DELAY  = 0
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

    
def forceAspect(axis,aspect=1):
    im = axis.get_images()
    extent =  im[0].get_extent()
    axis.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def define_color():

    startcolor = '#FDCC8A' 
    midcolor = '#FC8D59'
    endcolor = '#D7301F'

    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('own',[startcolor,midcolor,endcolor])
    plt.cm.register_cmap(cmap=cmap2)
    cmap2.set_under('#000000')
       

def plot_amp_weight(matrix,AP, x, savepath,name):
    
    
    diff = matrix[:,:]/matrix[0,:] # normalize to values with 0 shunt_weight
    noAP_idx = AP[:,:]<80
    diff[noAP_idx] = -2

    define_color()
    fig = plt.figure()
    ax = plt.subplot(111)    
    imgplot = plt.imshow(diff[:,:],vmin=-0,vmax=1,origin = 'lower',aspect = 'equal',interpolation="nearest")
    imgplot.set_cmap('own')

    plt.xticks(np.arange(0,100,10),np.arange(-5,5,1))
    plt.xlabel('delay [ms]')
    plt.yticks(np.arange(0,10,2),np.arange(0,0.1,0.02))
    plt.ylabel('weight [nS]')

    forceAspect(ax)
    cbar_ax = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    cb = fig.colorbar(imgplot, cax=cbar_ax)
    plt.savefig('%s/overview%s.eps'%(savepath,name))

def main():

    my_rawdata = {}

    condition = params['condition']    

    #inputs
    freq = params['Input']['freq'].value
    w_ee = params['Input']['wee'].value

    # Synapses
    pos = params['Synapse']['pos'].value
    shunt_pos = params['shunt']['shunt_pos'].value

    proximal_shunt_pos = params['shunt']['proximal_shunt_pos'].value
    distal_shunt_pos = params['shunt']['distal_shunt_pos'].value
    basal_shunt_pos = params['shunt']['basal_shunt_pos'].value
    proximal_shunt_compartment = params['shunt']['proximal_shunt_compartment']
    distal_shunt_compartment = params['shunt']['distal_shunt_compartment']
    basal_shunt_compartment = params['shunt']['basal_shunt_compartment']
    
    source = False

    delta_t = params['STDP']['delta_t'].value

    sim_params = params['sim']
    shunt_params = params['shunt']

    delay_start = params['shunt']['delay_start'].value
    delay_end = params['shunt']['delay_end'].value
    
    shunt_weight_range = np.arange(0.0,0.1,0.01)
    shunt_delay_range = np.arange(delay_start,delay_end,0.1)
    
    shunt_weight_range = np.arange(0.0,0.1,0.05)
    shunt_delay_range = np.arange(-1,1,1)


    AP = np.zeros((len(shunt_weight_range),len(shunt_delay_range)))

    distal_bAP = np.zeros((len(shunt_weight_range),len(shunt_delay_range)))
    distal_Ca = np.zeros((len(shunt_weight_range),len(shunt_delay_range)))
    distal_Ca_max = np.zeros((len(shunt_weight_range),len(shunt_delay_range)))

    oblique_bAP = np.zeros((len(shunt_weight_range),len(shunt_delay_range)))
    basal_bAP = np.zeros((len(shunt_weight_range),len(shunt_delay_range)))
    AP_time = np.zeros((len(shunt_weight_range),len(shunt_delay_range)))

    count = 1
    total_iterations = len(shunt_weight_range)*len(shunt_delay_range)
    
    for i, shunt_weight in enumerate(shunt_weight_range):
        for j, shunt_delay in enumerate(shunt_delay_range): 
            print "iteration %d from %d"%(count,total_iterations) 

            cell = Neuron()
            sim = Simulation(cell,sim_params)
            sim.dt = DT
            sim.v_init = -70
        
            # set dendritic location of interest
            if pos == 1:
                syn = h.ExpSynSTDP(cell.basal_main(0.5))
                weight = params['Synapse']['basal_weight'].value
            elif pos == 2:
                syn = h.ExpSynSTDP(cell.oblique_branch(0.5))
                weight = params['Synapse']['oblique_weight'].value
            elif pos == 3:
                syn = h.ExpSynSTDP(cell.branches[0](0.1))
                weight = params['Synapse']['distal_weight'].value
            else:
                syn = h.ExpSynSTDP(cell.soma(0.5))
                weight = 0
        
            interval = 1000.0/freq
            exstim = h.NetStim()
            exstim.number = NO_REPS
            exstim.interval = interval
            exstim.start = WARM_UP # reverse: +DELTA_T
            exstim.noise= 0
            exnc = h.NetCon(exstim,syn,0,0,weight)
        
            total_time = WARM_UP+NO_REPS*(1000.0/freq)+100
        
            sim.sim_time = total_time
        
            # dendritic excitation
            ns = h.NetStim()
            ns.number = NO_REPS
            ns.interval = 1000.0/freq
            ns.start = WARM_UP + delta_t - AP_DELAY # ms (most likely) start time of first spike
            ex_ex = h.List()
            nc_ex = h.List()
            for k in range(1,NO_EPSPs+1):
                pos = k * 0.1
                ex = neuron.h.Exp2Syn(cell.apical(pos))
                w = w_ee/NO_EPSPs
                ex.tau1 = 0.5 # ms rise time
                ex.tau2 = 2 # ms decay time
                ex.e = 0 # mV reversal p
                # excite postsynaptic cell
                nc = h.NetCon(ns,ex,1,0,w)
                ex_ex.append(ex)
                nc_ex.append(nc)
        
        
            # inhibition
            inh_delay = WARM_UP + delta_t - AP_DELAY + shunt_delay
        
            if shunt_pos == 0:
                print "no inhibition"
            elif shunt_pos == 1: # distal
                sim.set_Shunt(shunt_params, distal_shunt_pos, shunt_weight, distal_shunt_compartment,source,inh_delay,NO_REPS, interval)
            elif shunt_pos == 2: # proximal
                sim.set_Shunt(shunt_params, proximal_shunt_pos, shunt_weight, proximal_shunt_compartment,source,inh_delay,NO_REPS,interval)
            elif shunt_pos == 3: # basal
                sim.set_Shunt(shunt_params, basal_shunt_pos, shunt_weight,basal_shunt_compartment,source,inh_delay,NO_REPS,interval)                
            else:
                raise ValueError
        
        
            # recording
            trec = h.Vector()
            trec.record(h._ref_t)
            varec = h.Vector()
            varec.record(cell.ais(0.5)._ref_v)
            vrec = h.Vector()
            vrec.record(cell.soma(0.5)._ref_v)
            vinhrec = h.Vector()
            pos = 0.18*5.0
            vinhrec.record(cell.apical_prox(pos)._ref_v)
            vdrec = h.Vector()
            vdrec.record(cell.branches[0](0.5)._ref_v)
            vorec = h.Vector()
            vorec.record(cell.oblique_branch(0.9)._ref_v)
            vbrec = h.Vector()
            vbrec.record(cell.basal_main(0.9)._ref_v)
            cadrec = h.Vector()
            cadrec.record(cell.branches[0](0.5)._ref_ica)
            grec = h.Vector()
            grec.record(exnc._ref_weight[1])
        
            if not shunt_pos == 0:
                sim.set_synaptic_recording(switch=False,all=False)
        
            # run simulation
            sim.go()
        
            t = np.array(trec)
            v = np.array(vrec)
            va = np.array(varec)
            vinh = np.array(vinhrec)
            vd = np.array(vdrec)
            vo = np.array(vorec)
            vb = np.array(vbrec)
            cad = np.array(cadrec)
            g = np.array(grec)
            

            AP[i,j] = np.max(v)-(-75)
            distal_bAP[i,j] = np.max(vd)-(-75)
            distal_Ca[i,j] = np.sum(cad)
            distal_Ca_max[i,j] = np.max(np.abs(cad))
            oblique_bAP[i,j] = np.max(vo)-(-75)
            basal_bAP[i,j] = np.max(vb)-(-75)
            AP_time[i,j] = t[np.argmax(v[:1200])]
                        
            count += 1
            
    # plot
    plot_amp_weight(distal_bAP, AP, shunt_weight_range, savepath, 'distal_bAP')
    plot_amp_weight(distal_Ca, AP, shunt_weight_range, savepath, 'distal_Ca')
    plot_amp_weight(distal_Ca_max, AP, shunt_weight_range, path, 'distal_Ca_max')
    plot_amp_weight(oblique_bAP, AP, shunt_weight_range, path, 'oblique_bAP')
    plot_amp_weight(basal_bAP, AP, shunt_weight_range, path, 'basal_bAP')
        
    my_rawdata['t'] = t
    my_rawdata['v'] = v
    my_rawdata['va'] = va
    my_rawdata['vinh'] = vinh
    my_rawdata['vd'] = vd
    my_rawdata['vo'] = vo
    my_rawdata['vb'] = vb
    my_rawdata['cad'] = cad

    my_rawdata['g'] = g
    rawdata = {'raw_data': my_rawdata}


    return rawdata

if __name__ == '__main__':

    tables = main()

