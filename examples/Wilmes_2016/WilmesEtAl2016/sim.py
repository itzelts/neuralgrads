#!/usr/bin/env python

"""simulation"""

# _title_     : sim.py
# _author_     : Katharina Anna Wilmes
# _mail_     : katharina.anna.wilmes __at__ cms.hu-berlin.de


# --Imports--
import sys
import neuron
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import random

import nrn
import datetime
import random
import os
from neuron import h



# --load Neuron graphical user interface--
if not ( h('load_file("nrngui.hoc")')):
    print "Error, cannot open NEURON gui"


# general functions
def truncated_gauss(N, mu, sigma=0.05, a=0, b=2):
    """Return a N random numbers from a truncated (a,b) Gaussian distribution."""

    pos = np.zeros(N)
    i = 0
    while i<N:
        x = random.gauss(mu,sigma)
        if a <= x <= b:
            pos[i] = round(x,2)
            i += 1
    x = np.linspace(0,1,100)
    return pos

def gauss(N, mu=0, sigma=0.05):
    """Return a N random numbers from a truncated (a,b) Gaussian distribution."""

    pos = np.zeros(N)
    i = 0
    while i<N:
        x = random.gauss(mu,sigma)
        pos[i] = round(x,2)
        i += 1
    x = np.linspace(0,1,100)
    return pos

class Simulation(object):
    """
    Objects of this class control a simulation. Example of use:
    >>> cell = Cell()
    >>> sim = Simulation(cell)
    >>> sim.go()
    """
    def __init__(self, cell, params):
        self.cell = cell
        self.sim_time = params['sim_time'].value
        self.dt = params['dt'].value
        self.celsius = params['celsius'].value
        self.v_init = params['v_init'].value
        self.high_res = params['high_res']
        self.theta = params['theta'].value
        self.go_already = False
        self.SE = False
        self.vector = False
        self.shunt = h.List()
        self.nc = h.List()
        self.ns = h.List()
        self.v_rest = dict()
        self.syn = h.List()
        self.nc_syn = h.List()
        self.ns_syn = h.List()
        self.inhibition = h.List()
        self.nc_inh = h.List()
        self.ns_inh = h.List()
        self.excitation = h.List()
        self.nc_exc = h.List()
        self.ns_exc = h.List()
        self.stimlist = h.List()
        self.vlist = h.List()
        self.vtimelist = h.List()
        self.stimnumber = 0
        self.mu = []
        self.interneuron = h.List()


    def set_colored_noise_stim(self,params):
        noise = neuron.h.IClamp(params.noise.pos)
        noise.delay = params.noise.delay
        noise.dur = 1e9
        VecStim = h.Vector()  #the ramp vector: linearly increasing from 0 to I_max
        VecStim.play(noise._ref_amp, h.dt)  #playing the ramp vector into the iramp.amp variable




    def set_current(self, vector):
        """
        gets an array with the current that should be delivered by the electrode
        that is named stim
        """
        self.vector = True
        global vList
        timepoints = vector.size    # =self.sim_time/self.dt
        v = h.Vector(timepoints)
        v_time = h.Vector(timepoints)
        for i in range(timepoints):
            v.x[i] = vector[i] # set var to value in vector
        for i in range(timepoints):
            v_time.x[i] = i * self.sim_time / float(timepoints) # =i*self.dt
        self.vlist.append(v)
        self.vtimelist.append(v_time)

    def get_Ramp(self,params):
        """
        defines a stim ramp with duration (10) ms from 0 to max (1) nA,
        concatenated to zeros for sim_time-duration (40-10=30) ms
        """
        ramp = np.arange(0, params.max.value,params.max.value / (params.duration.value / self.dt))
        vector = np.concatenate((ramp, np.zeros(((self.sim_time - params.duration.value) / self.dt))))
        return vector


    def get_Alpha(self,params):
        """
        defines Alpha-function stimulus
        """

        x = np.arange(0, params.duration.value, self.dt)
        if self.SE == True:
            alpha = (self.cell.E + params.peak.value * (x / params.tau.value) * np.exp( - (x - params.tau.value) / params.tau.value))
            vector = np.concatenate((alpha,-70 * np.ones(((self.sim_time - params.duration.value) / self.dt))))
        else:
            alpha = params.max.value * (x / params.tau.value) * np.exp( - (x - params.tau.value) / params.tau.value)
            vector = np.concatenate((alpha, np.zeros(((self.sim_time - params.duration.value) / self.dt))))
        plt.figure()
        plt.plot(np.arange(0, self.sim_time, self.dt), vector)
        plt.show()
        return vector


    def set_Poisson(self,params, rate = None):
        print "set Poisson"
        if rate == None:
            rate = params.rate.value
        else:
            print rate
        syn_pos = params.pos.value
        g = params.g.value
        E = params.E.value
        tau1 = params.tau1.value
        tau2 = params.tau2.value
        number = params.number.value

        if params.location == 'soma' or params.type == 'excitation':
            synapse = neuron.h.Exp2Syn(self.cell.soma(0.5))
        elif params.location == 'apical':
            pos = (syn_pos / 2.0) + (syn_pos-0.5)
            synapse = neuron.h.Exp2Syn(self.cell.apical(pos))
        else:
            raise ValueError

        synapse.e = E
        synapse.tau1 = tau1
        synapse.tau2 = tau2


        ns = neuron.h.NetStim(0.5)
        ns.noise = 1
        ns.start = params.delay.value
        ns.number = number
        ns.interval = 1000.0/rate
        w = float(g)
        nc = h.NetCon(ns,synapse,1,1,w)

        self.syn.append(synapse)
        self.ns_syn.append(ns)
        self.nc_syn.append(nc)



    def set_distributed_shunt(self, params, shunt_pos, shunt_weight, compartment, number, pos_sigma, pattern, timing_sigma, delay, no_reps = 1, interval = None):

        shunt = h.List()
        nc = h.List()
        ns = h.List()
        pos = truncated_gauss(number,shunt_pos,pos_sigma,a=0,b=1) # params.number = number of shunts
        timing = gauss(number, 0, timing_sigma)

        pos.sort()
        for x in pos:
            if x <= 0.2:
                x = x * 5.0
                shunt.append(neuron.h.Exp2Syn(self.cell.apical_prox(x)))
            else:
                x = (x-0.2) * (1.0+(1.0/4.0))
                shunt.append(neuron.h.Exp2Syn(self.cell.apical(x)))
                sec = self.cell.apical.name()

        shunt_count = 0
        for s in shunt:
            s.tau1 = params.tau1.value # ms rise time
            s.tau2 = params.tau2.value # ms decay time
            s.e = params.e.value # mV reversal potential
            ns_i = neuron.h.NetStim(0.5)
            ns_i.number = 1 # (average) number of spikes
            if pattern == 'none':
                # for temporal patterns add delay to shunts at later pos
                ns_i.start = delay # ms (most likely) start time of first spike
            elif pattern == 'gauss':
                ns_i.start = delay + timing[shunt_count]
            elif pattern == 'random':
                ns_i.start = delay - 1 + (2.0 / number) * randi[shunt_count]
            elif pattern == 'sequence':
                ns_i.start = delay - 1 + (2.0 / number) * shunt_count
            else:
                ns_i.start = delay - 1 + (2.0 / number) * (number - shunt_count)
            w = (1.0 / number) * float(shunt_weight)
            ns.append(ns_i)
            nc.append(h.NetCon(ns_i,s,1,0,w))
            shunt_count += 1

        self.shunt = shunt
        self.ns = ns
        self.nc = nc

        return pos, timing


    def set_Shunt(self, params, shunt_pos, shunt_weight, compartment, source = False, delay=None, no_reps = 1, interval = None):

        """
        sets a single or a number of shunts at the defined shunt_pos (e.g. 0.3)
        of the shunt_compartment (e.g. "a", "o", or "b") with a particular shunt_weight (in uS)
        and a delay as specified in params.delay.value unless the function is called
        with delay.
        """
        # apical
        if compartment == 'a':
            if shunt_pos <= 0.2:
                pos = shunt_pos * 5.0
                print pos
                shunt = neuron.h.Exp2Syn(self.cell.apical_prox(pos))
                sec = self.cell.apical_prox.name()
                print sec
            else:
                pos = (shunt_pos-0.2) * (1.0+(1.0/4.0))
                shunt = neuron.h.Exp2Syn(self.cell.apical(pos))
                sec = self.cell.apical.name()
        # oblique
        elif compartment == 'o':
            shunt = neuron.h.Exp2Syn(self.cell.oblique_branch(shunt_pos-1))
            sec = self.cell.oblique_branch.name()
        # apical branch
        elif compartment == 'b':
            shunt = neuron.h.Exp2Syn(self.cell.branches[0](shunt_pos))
            sec = self.cell.branches[0].name()
        # basal
        elif compartment == 'basal':
            shunt = neuron.h.Exp2Syn(self.cell.basal_main(shunt_pos))
            sec = self.cell.basal_main.name()
        # basal branch
        elif compartment == 'basalb':
            shunt = neuron.h.Exp2Syn(self.cell.basal_branches[0](shunt_pos-1))
            sec = self.cell.basal_branches.name()
        else:
            shunt = neuron.h.Exp2Syn(self.cell.branches[1](shunt_pos-1))
        print params
        shunt.tau1 = params['tau1'].value
        shunt.tau2 = params['tau2'].value
        shunt.e = params['reversal'].value
        self.shunt_sec = sec
        ns = neuron.h.NetStim(0.5)
        ns.number = no_reps
        if interval is not None:
            ns.interval = interval
        w = float(shunt_weight)
        if delay == None: # if no value was provided
            ns.start = params.delay.value # ms (most likely) start time of first spike
        else:
            ns.start = delay
        # if inhibition is timed to activity of post-synaptic neuron
        if source:
            nc = h.NetCon(self.cell.soma(0.5)._ref_v, shunt, 1,1,w, sec = self.cell.soma)
        else:
            nc = h.NetCon(ns,shunt,1,0,w)

        self.shunt.append(shunt)
        # list makes it possible to have multiple shunts without overriding
        self.ns.append(ns)
        self.nc.append(nc)




    def set_synaptic_recording(self, switch = True, all=True):
        self.Inh_distalRec = neuron.h.Vector()
        self.nc[0].record(self.Inh_distalRec) # shunting inhibition from set_shunt
        if len(self.nc)>1:
            self.Inh_distalRec2 = neuron.h.Vector()
            self.nc[1].record(self.Inh_distalRec2) # shunting inhibition from set_shunt

        i = 1
        if switch:
            self.Inh2_distalRec = neuron.h.Vector()
            self.nc_syn[i].record(self.Inh2_distalRec)
            i +=1
        if all:
            self.ExcRec = neuron.h.Vector()
            self.InhRec = neuron.h.Vector()
            self.nc_syn[i].record(self.ExcRec)
            self.nc_syn[i+1].record(self.InhRec)

    def set_synaptic_current_rec(self):
        self.syn_current = neuron.h.Vector()
        self.syn_current.record(self.shunt[0]._ref_i)

    def set_synaptic_cond_rec(self):
        self.syn_cond = neuron.h.Vector()
        self.syn_cond.record(self.shunt[0]._ref_g)

    def set_rec_pos(self,rec_pos):
        self.rec_pos = rec_pos


    def set_highestres_recording(self):

        '''Record Time'''
        self.rec_t = neuron.h.Vector()
        self.rec_t.record(neuron.h._ref_t)

        '''record voltage for all compartments'''
        self.rec_va = neuron.h.Vector()
        self.rec_v = neuron.h.Vector()
        self.rec_v1 = neuron.h.Vector()
        self.rec_v2 = neuron.h.Vector()
        self.rec_v3 = neuron.h.Vector()
        self.rec_v4 = neuron.h.Vector()
        self.rec_v5 = neuron.h.Vector()
        self.rec_v6 = neuron.h.Vector()
        self.rec_v7 = neuron.h.Vector()
        self.rec_v8 = neuron.h.Vector()
        self.rec_v9 = neuron.h.Vector()
        self.rec_v10 = neuron.h.Vector()
        self.rec_v11 = neuron.h.Vector()
        self.rec_v12 = neuron.h.Vector()
        self.rec_v13 = neuron.h.Vector()
        self.rec_v14 = neuron.h.Vector()
        self.rec_v15 = neuron.h.Vector()
        self.rec_v16 = neuron.h.Vector()
        self.rec_v17 = neuron.h.Vector()
        self.rec_v18 = neuron.h.Vector()
        self.rec_v19 = neuron.h.Vector()
        self.rec_v20 = neuron.h.Vector()
        self.rec_v21 = neuron.h.Vector()
        self.rec_v22 = neuron.h.Vector()
        self.rec_v23 = neuron.h.Vector()
        self.rec_v24 = neuron.h.Vector()
        self.rec_v25 = neuron.h.Vector()
        self.rec_v26 = neuron.h.Vector()
        self.rec_v27 = neuron.h.Vector()
        self.rec_v28 = neuron.h.Vector()
        self.rec_v29 = neuron.h.Vector()
        self.rec_v30 = neuron.h.Vector()
        self.rec_v31 = neuron.h.Vector()
        self.rec_v32 = neuron.h.Vector()
        self.rec_v33 = neuron.h.Vector()
        self.rec_v34 = neuron.h.Vector()
        self.rec_v35 = neuron.h.Vector()
        self.rec_v36 = neuron.h.Vector()
        self.rec_v37 = neuron.h.Vector()
        self.rec_v.record(self.cell.soma(0.5)._ref_v)
        self.rec_va.record(self.cell.ais(0.5)._ref_v)
        '''record apical and oblique if existent'''
        self.rec_v1.record(self.cell.apical_prox(0.3)._ref_v)
        self.rec_v2.record(self.cell.apical_prox(0.6)._ref_v)
        self.rec_v3.record(self.cell.apical_prox(0.9)._ref_v)
        self.rec_v4.record(self.cell.apical(0.15)._ref_v)
        self.rec_v5.record(self.cell.apical(0.3)._ref_v)
        self.rec_v6.record(self.cell.apical(0.45)._ref_v)
        self.rec_v7.record(self.cell.apical(0.6)._ref_v)
        self.rec_v8.record(self.cell.apical(0.75)._ref_v)
        self.rec_v9.record(self.cell.apical(0.9)._ref_v)


        self.rec_vo1 = neuron.h.Vector()
        self.rec_vo2 = neuron.h.Vector()
        self.rec_vo3 = neuron.h.Vector()
        self.rec_vo4 = neuron.h.Vector()
        self.rec_vo5 = neuron.h.Vector()
        self.rec_vo6 = neuron.h.Vector()
        self.rec_vo7 = neuron.h.Vector()
        self.rec_vo8 = neuron.h.Vector()
        self.rec_vo9 = neuron.h.Vector()
        self.rec_vo1.record(self.cell.oblique_branch(0.1)._ref_v)
        self.rec_vo2.record(self.cell.oblique_branch(0.2)._ref_v)
        self.rec_vo3.record(self.cell.oblique_branch(0.3)._ref_v)
        self.rec_vo4.record(self.cell.oblique_branch(0.4)._ref_v)
        self.rec_vo5.record(self.cell.oblique_branch(0.5)._ref_v)
        self.rec_vo6.record(self.cell.oblique_branch(0.6)._ref_v)
        self.rec_vo7.record(self.cell.oblique_branch(0.7)._ref_v)
        self.rec_vo8.record(self.cell.oblique_branch(0.8)._ref_v)
        self.rec_vo9.record(self.cell.oblique_branch(0.9)._ref_v)

        self.rec_vbasal1 = neuron.h.Vector()
        self.rec_vbasal2 = neuron.h.Vector()
        self.rec_vbasal3 = neuron.h.Vector()
        self.rec_vbasal4 = neuron.h.Vector()
        self.rec_vbasal5 = neuron.h.Vector()
        self.rec_vbasal6 = neuron.h.Vector()
        self.rec_vbasal7 = neuron.h.Vector()
        self.rec_vbasal8 = neuron.h.Vector()
        self.rec_vbasal9 = neuron.h.Vector()
        self.rec_vbasal10 = neuron.h.Vector()
        self.rec_vbasal11 = neuron.h.Vector()
        self.rec_vbasal12 = neuron.h.Vector()
        self.rec_vbasal13 = neuron.h.Vector()
        self.rec_vbasal14 = neuron.h.Vector()
        self.rec_vbasal1.record(self.cell.basal_main(0.1)._ref_v)
        self.rec_vbasal2.record(self.cell.basal_main(0.2)._ref_v)
        self.rec_vbasal3.record(self.cell.basal_main(0.3)._ref_v)
        self.rec_vbasal4.record(self.cell.basal_main(0.4)._ref_v)
        self.rec_vbasal5.record(self.cell.basal_main(0.5)._ref_v)
        self.rec_vbasal6.record(self.cell.basal_main(0.6)._ref_v)
        self.rec_vbasal7.record(self.cell.basal_main(0.7)._ref_v)
        self.rec_vbasal8.record(self.cell.basal_main(0.8)._ref_v)
        self.rec_vbasal9.record(self.cell.basal_main(0.9)._ref_v)
        self.rec_vbasal10.record(self.cell.basal_branches[0](0.1)._ref_v)
        self.rec_vbasal11.record(self.cell.basal_branches[0](0.3)._ref_v)
        self.rec_vbasal12.record(self.cell.basal_branches[0](0.5)._ref_v)
        self.rec_vbasal13.record(self.cell.basal_branches[0](0.7)._ref_v)
        self.rec_vbasal14.record(self.cell.basal_branches[0](0.9)._ref_v)


        '''record branches'''
        self.rec_v20.record(self.cell.branches[0](0.05)._ref_v)
        self.rec_v21.record(self.cell.branches[0](0.1)._ref_v)
        self.rec_v22.record(self.cell.branches[0](0.15)._ref_v)
        self.rec_v23.record(self.cell.branches[0](0.2)._ref_v)
        self.rec_v24.record(self.cell.branches[0](0.25)._ref_v)
        self.rec_v25.record(self.cell.branches[0](0.3)._ref_v)
        self.rec_v26.record(self.cell.branches[0](0.35)._ref_v)
        self.rec_v27.record(self.cell.branches[0](0.4)._ref_v)
        self.rec_v28.record(self.cell.branches[0](0.45)._ref_v)
        self.rec_v29.record(self.cell.branches[0](0.5)._ref_v)
        self.rec_v30.record(self.cell.branches[0](0.55)._ref_v)
        self.rec_v31.record(self.cell.branches[0](0.6)._ref_v)
        self.rec_v32.record(self.cell.branches[0](0.65)._ref_v)
        self.rec_v33.record(self.cell.branches[0](0.7)._ref_v)
        self.rec_v34.record(self.cell.branches[0](0.75)._ref_v)
        self.rec_v35.record(self.cell.branches[0](0.8)._ref_v)
        self.rec_v36.record(self.cell.branches[0](0.85)._ref_v)
        self.rec_v37.record(self.cell.branches[0](0.9)._ref_v)

        self.rec_distal = neuron.h.Vector()
        self.rec_distal.record(self.cell.second_branches[0](0.5)._ref_v)



        if self.cell.ifca:

            self.rec_icaL = neuron.h.Vector()
            self.rec_icaL1 = neuron.h.Vector()
            self.rec_icaL2 = neuron.h.Vector()
            self.rec_icaL3 = neuron.h.Vector()
            self.rec_icaL4 = neuron.h.Vector()
            self.rec_icaL5 = neuron.h.Vector()
            self.rec_icaL6 = neuron.h.Vector()
            self.rec_icaL7 = neuron.h.Vector()
            self.rec_icaL8 = neuron.h.Vector()
            self.rec_icaL9 = neuron.h.Vector()
            self.rec_icaL10 = neuron.h.Vector()
            self.rec_icaL11 = neuron.h.Vector()
            self.rec_icaL1.record(self.cell.apical_prox(0.2)._ref_ica)
            self.rec_icaL2.record(self.cell.apical_prox(0.6)._ref_ica)
            self.rec_icaL3.record(self.cell.apical_prox(1)._ref_ica)
            self.rec_icaL4.record(self.cell.apical(0.4)._ref_ica)
            self.rec_icaL5.record(self.cell.apical(0.8)._ref_ica)
            self.rec_icaL6.record(self.cell.branches[0](0.1)._ref_ica)
            self.rec_icaL7.record(self.cell.branches[0](0.5)._ref_ica)
            self.rec_icaL8.record(self.cell.branches[0](0.9)._ref_ica)
            self.rec_icaL9.record(self.cell.oblique_branch(0.1)._ref_ica)
            self.rec_icaL10.record(self.cell.oblique_branch(0.3)._ref_ica)
            self.rec_icaL11.record(self.cell.oblique_branch(0.5)._ref_ica)


            self.rec_ina0 = neuron.h.Vector()
            self.rec_ina1 = neuron.h.Vector()
            self.rec_ina2 = neuron.h.Vector()
            self.rec_ina3 = neuron.h.Vector()
            self.rec_ina4 = neuron.h.Vector()
            self.rec_ina5 = neuron.h.Vector()
            self.rec_ina6 = neuron.h.Vector()
            self.rec_ina7 = neuron.h.Vector()
            self.rec_ina8 = neuron.h.Vector()
            self.rec_ina9 = neuron.h.Vector()
            self.rec_ina10 = neuron.h.Vector()
            self.rec_ina11 = neuron.h.Vector()
            self.rec_ina0.record(self.cell.ais(0.5)._ref_ina)
            self.rec_ina1.record(self.cell.apical_prox(0.2)._ref_ina)
            self.rec_ina2.record(self.cell.apical_prox(0.6)._ref_ina)
            self.rec_ina3.record(self.cell.apical_prox(0.9)._ref_ina)
            self.rec_ina4.record(self.cell.apical(0.4)._ref_ina)
            self.rec_ina5.record(self.cell.apical(0.8)._ref_ina)
            self.rec_ina6.record(self.cell.branches[0](0.1)._ref_ina)
            self.rec_ina7.record(self.cell.branches[0](0.5)._ref_ina)
            self.rec_ina8.record(self.cell.branches[0](0.9)._ref_ina)
            self.rec_ina9.record(self.cell.oblique_branch(0.1)._ref_ina)
            self.rec_ina10.record(self.cell.oblique_branch(0.3)._ref_ina)
            self.rec_ina11.record(self.cell.oblique_branch(0.5)._ref_ina)


    def get_spike_dict(self):
        spike_dict = spiketrainutil.netconvecs_to_dict(self.t_vec, self.id_vec)
        return spike_dict

    def get_spike_times(self,voltage):
        voltage = voltage - (-70)
        above_threshold = voltage > 50 # gives a vector of zeros and ones
        threshold_crossings = np.diff(above_threshold) # gives ones when 1 changes to 0 or reverse
        strokes = np.nonzero(threshold_crossings)
        upstrokes = strokes[0][::2]
        print upstrokes
        downstrokes = strokes[0][1::2]
        spike_times = np.zeros(len(downstrokes))
        for i in range(len(downstrokes)):
            spike_time = upstrokes[i] + np.argmax(voltage[upstrokes[i]:downstrokes[i]])
            spike_times[i] = int(spike_time)
        spike_times = spike_times.astype(int)
        spike_times = spike_times * self.dt
        print "spike_times: " + str(spike_times)
        return spike_times


    def get_idx(self,time):
        "transforms time [ms] into timestep"
        idx = time*np.int(1/self.dt)
        return idx

    def get_spike_train(self,voltage):
        st = self.get_spike_times(voltage)
        spike_train = np.zeros(len(voltage))
        spike_train[st] = 1
        return spike_train


    def get_rate(self,start,end):
        idx_start = self.get_idx(start)
        idx_end = self.get_idx(end)
        voltage = np.array(self.rec_v) - (-70) # I used to measure this at the ais
        voltage = voltage[idx_start:idx_end]
        above_threshold = voltage > 50 # gives a vector of zeros and ones
        threshold_crossings = np.diff(above_threshold) # gives ones when 1 changes to 0 or reverse
        number_of_spikes = np.shape(np.nonzero(threshold_crossings))[1]*0.5
        if number_of_spikes == 0:
            return number_of_spikes, 0
        else:
            return number_of_spikes, (number_of_spikes * 1000.0) / (end-start)



    def get_ISI(self,voltage,start,end):
        idx_start = self.get_idx(start)
        idx_end = self.get_idx(end)
        spiketimes = self.get_spike_times(voltage[idx_start:idx_end])
        ISI = np.diff(spiketimes)
        return ISI


    def get_bursts(self, voltage, start, end):
        idx_start = self.get_idx(start)
        idx_end = self.get_idx(end)
        spiketimes = self.get_spike_times(voltage[idx_start:idx_end])
        spike_timings = spiketimes # spike_timings have unit mst

        ISI = np.diff(spike_timings)
        small_ISI = np.zeros(len(ISI))
        small_ISI[ISI<25] = 1
        small_ISI = np.append(np.array(0),small_ISI)

        ISI_boundaries = np.diff(small_ISI)
        idx = np.nonzero(ISI_boundaries)
        try:
            ISIstart = np.nonzero(ISI_boundaries==1)[0]
            ISIend = np.nonzero(ISI_boundaries==-1)[0]
            if not np.shape(ISIstart)[0] == np.shape(ISIend)[0]: # if there is not an endpoint for a startpoint
                ISIend = np.append(ISIend, np.array(len(spike_timings)-1))
                # add last spike as burst end, because there was no single spike after the last burst

        except IndexError:
            print "IndexError excepted in sim.get_bursts()"
        realbursts = (ISIend-ISIstart)>1
        ISIstart = ISIstart[realbursts]
        ISIend = ISIend[realbursts]
        print ISIstart
        print ISIend
        burststart = spike_timings[ISIstart]
        burstend = spike_timings[ISIend]
        print burststart
        print burstend
        print "end get_bursts"
        return ISIstart, ISIend, burststart, burstend

    def get_burst_stats(self, voltage, start, end):
        ISIstart, ISIend, burststart, burstend = self.get_bursts(voltage, start, end)
        diff = ISIend - ISIstart
        print "diff: " + str(diff)
        burstnumber = np.shape(np.nonzero(diff>1))[1]
        spikes_in_bursts = np.sum(diff[np.nonzero(diff>1)]) + burstnumber # spikes = ISI + 1
        print "burstnumber: " + str(burstnumber)
        print "spikes_in_bursts: " + str(spikes_in_bursts)
        if burstnumber == 0:
            burstrate = 0
        else:
            burstrate = (burstnumber * 1000.0) / (end-start)
        if spikes_in_bursts == 0:
            spikes_in_bursts_rate = 0
        else:
            spikes_in_bursts_rate = (spikes_in_bursts * 1000.0) / (end-start)

        return burstrate, spikes_in_bursts, spikes_in_bursts_rate


    def plot_bursts(self, voltage, start, end):
        ISIstart, ISIend, burststart, burstend = self.get_bursts(voltage, start, end)
        idx_burststart = self.get_idx(burststart)
        idx_burstend = self.get_idx(burstend)
        idx_start = self.get_idx(start)
        idx_end = self.get_idx(end)
        voltage = voltage[idx_start:idx_end]
        plot = Plot(self, params = P.plot)
        for i in np.arange(0,np.shape(burststart)[0]):
            print "voltage"
            print voltage[idx_burststart[i]-80:idx_burstend[i]+80]
            plot.voltage(voltage[idx_burststart[i]-80:idx_burstend[i]+80],'burst%d'%i)


    def plot_spikes(self,spikes):
        spike_list = list()
        for value in spikes.itervalues():
            print value
            spike_list.append(value)
        sp = spikeplot.SpikePlot(fig_name = "/extra/wilmes/figures/spikeplot.png")
        print spike_list
        sp.plot_spikes(spike_list, label=None, draw=True, savefig=True, cell_offset=0)


    def set_basal_recorders(self):
        '''record basal voltage for all compartments'''
        self.basallist = neuron.h.List()
        for seg in self.cell.basal_main:
            basal = neuron.h.Vector() # int(tstopms/self.timeres_python+1))
            basal.record(seg._ref_v)
            self.basallist.append(basal)

    def get_recording(self):
        self.recording = np.array((self.rec_v,self.rec_v1,self.rec_v2,self.rec_v3,
        self.rec_v4,self.rec_v5,self.rec_v6,self.rec_v7,self.rec_v8,self.rec_v9,self.rec_v10))
        return self.recording

    def get_otherbranch_recording(self):
        self.recording = np.array((self.rec_v,self.rec_v1,self.rec_v1a,self.rec_v2,
        self.rec_v2a,self.rec_v3,self.rec_v3a,self.rec_v4,self.rec_v4a,self.rec_v5,
        self.rec_v6o,self.rec_v7o,self.rec_v8o,self.rec_v9o,self.rec_v10o))
        return self.recording


    def get_highestres_recording(self):
        self.recording = np.array((self.rec_va, self.rec_v,self.rec_v1,self.rec_v2,self.rec_v3,
        self.rec_v4,self.rec_v5,self.rec_v6,self.rec_v7,self.rec_v8,self.rec_v9,
        self.rec_v20,self.rec_v21,
        self.rec_v22,self.rec_v23,self.rec_v24,self.rec_v25,self.rec_v26,self.rec_v27,
        self.rec_v28,self.rec_v29,self.rec_v30,self.rec_v31,self.rec_v32,self.rec_v33,
        self.rec_v34, self.rec_v35,self.rec_v36,self.rec_v37))

        return self.recording

    def get_oblique_recording(self):
        self.oblique_rec = np.array((self.rec_va, self.rec_v,self.rec_v2,
        self.rec_v4,self.rec_v6,self.rec_v8,self.rec_v9,self.rec_vo1,
        self.rec_vo2,self.rec_vo3, self.rec_vo4,self.rec_vo5,self.rec_vo6,
        self.rec_vo7,self.rec_vo8,self.rec_vo9))
        return self.oblique_rec

    def get_highres_oblique_recording(self):
        self.oblique_rec = np.array((self.rec_va, self.rec_v,self.rec_v1,self.rec_v2,self.rec_v3,self.rec_vo1,
        self.rec_vo2,self.rec_vo3, self.rec_vo4,self.rec_vo5,self.rec_vo6,
        self.rec_vo7,self.rec_vo8,self.rec_vo9))
        return self.oblique_rec


    def get_basal_recording(self):
        self.basal_rec = np.array((self.rec_va, self.rec_v, self.rec_vbasal1, self.rec_vbasal2,self.rec_vbasal3,
        self.rec_vbasal4,self.rec_vbasal5,self.rec_vbasal6,self.rec_vbasal7,self.rec_vbasal8,
        self.rec_vbasal9,self.rec_vbasal10, self.rec_vbasal11,self.rec_vbasal12,self.rec_vbasal13,
        self.rec_vbasal14))
        return self.basal_rec

    def get_basal_recording1(self):
        return self.basallist

    def get_ais_ina(self):
        return np.array((self.rec_ina0))

    def get_ina(self):
        self.ina = np.array((self.rec_ina1,self.rec_ina2,self.rec_ina3,
        self.rec_ina4,self.rec_ina5,self.rec_ina6,self.rec_ina7, self.rec_ina8,
        self.rec_ina9,self.rec_ina10,self.rec_ina11))
        return self.ina

    def get_ica(self):
        self.ical = np.array((self.rec_icaL1, self.rec_icaL2, self.rec_icaL3,
        self.rec_icaL4, self.rec_icaL5, self.rec_icaL6, self.rec_icaL7, self.rec_icaL8, self.rec_icaL9, self.rec_icaL10, self.rec_icaL11))
        return self.ical

    def get_ca_current(self,start,end):
        idx_start = self.get_idx(start)
        idx_end = self.get_idx(end)
        ca_current = np.array(self.rec_icaL7)
        return ca_current[idx_start:idx_end]


    def get_ica_simple(self):
        self.ica_simple = np.array((self.rec_icaL1, self.rec_icaL2, self.rec_icaL3,
        self.rec_icaL4, self.rec_icaL5, self.rec_icaL6, self.rec_icaL7, self.rec_icaL8,
        self.rec_icaL9, self.rec_icaL10, self.rec_icaL11))
        return self.ica_simple

    def get_time(self):
        return self.rec_t

    def get_stim_current(self):
        return self.rec_stim_i

    def get_stim_voltage(self):
        return self.rec_stim_vc

    def get_shunt_current(self):
        return self.rec_i

    def get_EPSC(self):
        return np.array((self.rec_EPSC))

    def get_shunt_conductance(self):
        return self.rec_g

    def get_distal_rec(self):
        return np.array((self.rec_distal))


    def get_synaptic_recording(self,kind):
        if kind == 'dendritic_inh':
            return np.array((self.Inh_distalRec))
        elif kind == 'dendritic_inh2':
            return np.array((self.Inh2_distalRec))
        elif kind == 'exc':
            return np.array((self.ExcRec))
        elif kind == 'inh':
            return np.array((self.InhRec))
        else:
            return ValueError

    def get_synaptic_current_rec(self):
        return np.array(self.syn_current)

    def get_synaptic_cond_rec(self):
        return np.array(self.syn_cond)

    def go(self, sim_time=None):
        #if self.high_res:
        #    self.set_highestres_recording()
        #    self.set_basal_recorders()
        #else:
        #    self.set_recording()
        neuron.h.dt = self.dt
        neuron.h.celsius = self.celsius

        if self.vector == True:
            if self.SE:
                self.v.play(self.stim._ref_amp1, self.v_time, 1)
            else:
                for i in range(self.stimnumber):
                    print "stimnumber: %d"%self.stimnumber
                    self.vlist[i].play(self.stimlist[i]._ref_amp, self.vtimelist[i], 1)
                    outime = np.array(self.vtimelist[i])
                    oustim = np.array(self.stimlist[i])
                    ouv = np.array(self.vlist[i])
                    print "ou"
                    print outime[::100]
                    print oustim[::100]
                    print ouv[::100]
                    print self.mu
        neuron.init()
        neuron.h.finitialize(self.v_init)

        h.fcurrent()


        neuron.run(self.sim_time)
        self.go_already = True
