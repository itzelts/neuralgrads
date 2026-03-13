#!/usr/bin/env python


"""config-file for feedforward circuit in Figure 7"""

# _title_     : config_circuit.py
# _author_     : Katharina Anna Wilmes
# _mail_     : katha __at__ bccn-berlin.de


# --Imports--
import os
import gc
import sys
import pickle
import h5py
from time import time
import datetime
import numpy as np

from NeuroTools.parameters import Parameter



#param_ranges = {
#   'Input':
#       {
#       'freq': ParameterArray([10]),
#        'w_ie': ParameterArray([0.0,0.05]),
#       },
#   'Synapse':
#       {
#        'EPSP':ParameterArray(['apical']), # 'apical' 'varying'
#        'inh_pos':ParameterArray([0.9]),
#        'inh_type':ParameterArray([0]), # 0 hh2, 1 hh3
#        'interneuron_target':ParameterArray([0]) #0 proximal, 1 distal
#
#       },
#   }

params = {
    'visual': 'figure',
    'results_file': 'pairingscenarios',
    'Input':
        {
        'freq': Parameter(10),
        'w_ei': Parameter(0.3),
        'w_ii':Parameter(0.0),
        'w_ee':Parameter(0.02)
        },
    'Synapse':
        {
        'pos': Parameter(3),
        'EPSP_pos':Parameter(0.7),
        'timing_sigma':Parameter(0),
        'distal_weight':Parameter(0.008),
        'oblique_weight':Parameter(0.001),
        'basal_weight':Parameter(0.005),
        },
    'shunt':
        {
        'reversal':Parameter(-73),
        'shunt_pos': Parameter(2),#0
        'tau1':Parameter(0.5),
        'tau2':Parameter(5),
        'distal_shunt_pos':Parameter(0.9),
        'proximal_shunt_pos':Parameter(0.18),
        'basal_shunt_pos':Parameter(0.3),
        'distal_shunt_compartment':'a',
        'proximal_shunt_compartment':'a',
        'basal_shunt_compartment':'basal',
        'shunt_pattern':'gauss',
        'shunt_number':Parameter(0),
        'shunt_delay':Parameter(1),
        'shunt_sigma':Parameter(0.05),
        'inh_type':Parameter(0),
        'interneuron_target':Parameter(0),
        'inh_pos':Parameter(0.9),
        },
    'cell':
       {
       'gsca' : Parameter(1.5),
       'git2' : Parameter(0.005),
       'gbar_kca' : Parameter(2.5),
       },
    'sim':
        {
        'duration' : Parameter(1,'second'),
        'v_init': Parameter(-70,'mV'),
        'dt': Parameter(0.1,'ms'),
        'sim_time':Parameter(100,'ms'), 
        'celsius': Parameter(30,'C'),
        'high_res': True,
        'theta':Parameter(1,'1/ms')
        },
    'plot':
        {
        'version':Parameter(1),
        'path': os.path.expanduser("/scratch/kwilmes/Project1")
        }
}
