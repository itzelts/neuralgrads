#!/usr/bin/env python

"""simple neuron model with soma and branching dendrite"""

# _title_     : neuronmodel.py
# _author_     : Katharina Anna Wilmes
# _mail_     : katharina.anna.wilmes __at__ cms.hu-berlin.de


# --Imports--
import neuron
import matplotlib.pyplot as plt
import numpy as np
import nrn
import datetime
from neuron import h, gui # adding gui makes main menu appear
from config_neuronmodel import *



# --load Neuron graphical user interface--
if not ( h('load_file("nrngui.hoc")')):
    print "Error, cannot open NEURON gui"



def make_compartment(length=150, diameter=1, nseg=5, Ra= 150, cm = 0.75):
    """
    Returns a compartment.
    """
    compartment = neuron.h.Section()
    compartment.L = length
    compartment.diam = diameter
    compartment.nseg = nseg
    compartment.Ra = Ra
    compartment.cm = cm
    return compartment


def fromtodistance(origin_segment, to_segment):
    h.distance(0, origin_segment.x, sec=origin_segment.sec)
    return h.distance(to_segment.x, sec=to_segment.sec)


class BiophysicsError(Exception):
    pass


class ShapeError(Exception):
    pass


class Neuron(object):
    """
    This class will produce Neuron objects with a soma with an apical 
    dendrite, a basal dendrite, and an oblique dendrite extending from 
    the apical one
    """


    def __init__(self):
        # morphology
        self.a_diam = params['Neuron']['a_diam'].value
        self.s_diam = params['Neuron']['s_diam'].value        
        self.d_diam = params['Neuron']['d_diam'].value
        self.d_length = params['Neuron']['d_length'].value
        self.apical_length = params['Neuron']['apical_length'].value
        self.nseg = params['Neuron']['n_seg'].value

        # passive properties
        self.Rm = params['Neuron']['R_m'].value 
        self.E = params['Neuron']['E_leak'].value 
        self.Ra = params['Neuron']['R_a'].value
        self.cm = params['Neuron']['C_m'].value
        self.gp = 1/float(self.Rm)

        # active conductances
        self.ena = params['Neuron']['E_Na'].value
        self.ek = params['Neuron']['E_K'].value
        self.gna = params['Neuron']['g_Na'].value
        self.gk = params['Neuron']['g_K'].value
        self.gk_kap = params['Neuron']['g_KA'].value  
        self.slope = params['Neuron']['slope_KA'].value
        self.gna_ais = params['Neuron']['g_Na_ais'].value
        self.gna_ais_shifted = params['Neuron']['g_Na_ais_shifted'].value   
        
        # calcium
        self.eca = params['Neuron']['E_Ca'].value
        self.git2 = params['Neuron']['git2'].value
        self.gsca = params['Neuron']['gsca'].value
        self.gbar_kca = params['Neuron']['g_KCa'].value
        self.ifca = params['Neuron']['ifca']

        self.na_vshift = params['Neuron']['ifshift'] 
        self.ais_vshift = 10
        self.dend_vshift = params['Neuron']['dend_vshift'].value


        """ creating compartments"""
        # axon inital segment
        self.ais = make_compartment(self.a_diam,self.a_diam*1.5,1)

        # soma
        self.soma = make_compartment(self.s_diam, self.s_diam,1)
        self.soma.connect(self.ais,1,0)

        # oblique branch
        self.oblique_branch = make_compartment(self.d_length, (1/5.0)*self.d_diam,self.nseg)

        # apical
        self.apical_prox = make_compartment(self.apical_length/5.0,self.d_diam,(self.nseg/5)+1)
        self.apical = make_compartment(4.0*self.apical_length/5.0,self.d_diam,(4*self.nseg/5)+1)
        self.apical_prox.connect(self.soma,1,0)
        self.apical.connect(self.apical_prox,1,0)
        
        self.oblique_branch.connect(self.apical_prox,1,0)

        # basal with branches
        self.basal_main = make_compartment(self.d_length/2.0,self.d_diam/2.0,self.nseg)
        self.basal_main.connect(self.soma,0,0)
        self.basal_branches = list()
        for i in range(2):
            branch = make_compartment(self.d_length/2.0, (2/3.0)*(self.d_diam/2.0),self.nseg)
            branch.connect(self.basal_main,1,0)
            self.basal_branches.append(branch)
        self.second_basal_branches = list()
        for i in range(4):
            branch = make_compartment(self.d_length/2.0, (2/3.0)*(2/3.0)*(self.d_diam/2.0),self.nseg)
            branch.connect(self.basal_branches[i%2],1,0)
            self.second_basal_branches.append(branch)

        # apical tuft
        self.branches = list()
        for i in range(2):
            branch = make_compartment(self.d_length, (2/3.0)*self.d_diam,self.nseg)
            branch.connect(self.apical,1,0)
            self.branches.append(branch)
        self.second_branches = list()
        for i in range(4):
            branch = make_compartment(self.d_length, (2/3.0)*(2/3.0)*self.d_diam,self.nseg)
            branch.connect(self.branches[i%2],1,0)
            self.second_branches.append(branch)

        # initialize parameters
        self.set_passive_parameters(self.gp, self.E, self.Ra)
        self.set_hh_parameters(self.ena, self.ek, self.gna, self.gk)
        self.set_kap_parameters(gkapbar = self.gk_kap, Ekap = self.ek)

        if self.ifca:
            print "ca"
            self.set_ca_parameters(gsca = self.gsca, git2 = self.git2,
            gkca = self.gbar_kca, eca = self.eca)


    def set_passive_parameters(self, gp, E, rho):
        for sec in neuron.h.allsec():
            sec.Ra = rho
            sec.insert("pas")
            for seg in sec:
                seg.pas.g = gp


    def set_hh_parameters(self, Ena, Ek, gnabar, gkbar):
        count = 0
        for sec in neuron.h.allsec():
                count += 1
                sec.insert('na3dend')
                h.vshift_na3dend = self.dend_vshift
                sec.insert('kdr')
                for seg in sec:
                    seg.gbar_na3dend = gnabar
                    seg.gkdrbar_kdr = gkbar
                    seg.ena = Ena
                    seg.ek = Ek

        if self.na_vshift:
            self.ais.gbar_na3dend = 0
            self.ais.insert('na3')
            self.ais.gbar_na3 = self.gna_ais
            self.ais.insert('na3shifted')
            h.vshift_na3shifted = self.ais_vshift
            self.ais.gbar_na3shifted = self.gna_ais_shifted
        elif not gnabar == 0:
            self.ais.gbar_na3 = self.gna_ais
        else:
            print 'no Na in ais'


    def set_ca_parameters(self, gsca, git2, gkca, eca):
        # h.distance(sec=self.soma, seg=0) # assigning seg=0 was ignored 
        h.distance(sec=self.soma)
        for sec in neuron.h.allsec():
                sec.insert('sca')
                sec.insert('cad2')
                sec.insert('kca')
                for seg in sec:
                    seg.eca = eca
                    if not sec == self.soma:
                        sec.insert('it2')
                        dist = fromtodistance(self.soma(0.5),seg)
                        if ((not sec == self.oblique_branch) and (dist > 500 and dist < 750) and (not sec in self.basal_branches)
                        and (not sec in self.second_basal_branches)):
                            seg.gcabar_it2 = git2 
                            seg.gbar_sca = gsca * 3 
                            seg.gbar_kca = gkca 
                        else:
                            seg.gcabar_it2 = 0
                            seg.gbar_sca = gsca 
                            seg.gbar_kca = gkca 

                    else:
                        for seg in sec:
                            seg.gbar_sca = gsca * 2 
                            seg.gbar_kca = gkca * 2 



    def set_kap_parameters(self, gkapbar, Ekap):
        #h.distance(sec=self.soma, seg=0) # assigning seg=0 was ignored 
        h.distance(sec=self.soma)
        for sec in neuron.h.allsec():
            sec.insert('kap')
            if not sec == self.soma:
                for seg in sec:
                    dist = fromtodistance(self.soma(0.5),seg)
                    if dist > 500:
                        dist = 500
                    seg.gkabar_kap = gkapbar*(1 + dist / (500 / self.slope))
                    seg.ek = Ekap
            else:
                self.soma.gkabar_kap = gkapbar
