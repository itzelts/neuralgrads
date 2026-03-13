#!/usr/bin/env python


"""config-file for pairingprotocol.py, to create data for Figure 6"""

# _title_     : config_pairing.py
# _author_     : Katharina Anna Wilmes
# _mail_     : katha __at__ bccn-berlin.de


# --Imports--
import os
import gc
import sys
import pickle
import h5py
from time import time
import numpy as np
import datetime

from NeuroTools.parameters import Parameter



#param_ranges = {
#   'Input':
#       {
#       'freq': ParameterArray([1]) # Hz
#       },
#   'Synapse':
#       {
#        'pos': ParameterArray([1,2,3]), # 1 basal, 2 oblique, 3 distal
#		'shunt_pos': ParameterArray([0,1,2,3]), #0 no 1 distal 2 proximal 3 basal
#        'distal_shunt_weight':ParameterArray([0.05]),
#        'proximal_shunt_weight':ParameterArray([0.05]),
#        'basal_shunt_weight':ParameterArray([0.05]), #0.1
#       },
#   'STDP':
#       {
#       'delta_t': ParameterArray((np.arange(-20,21,1))),
#       }
#   }

params = {
    'visual': 'figure',
    'results_file': 'pairingscenarios',
    'set_me':
        {
        'pos': Parameter(1), # 1 basal, 2 oblique, 3 distal
        'shunt_pos': Parameter(0), # 0 no, 1 distal, 2 proximal, 3 basal
        },
    'Input':
        {
        'freq': Parameter(10) # set to 1
        },
    'Synapse':
        {
        'shunt_reversal':Parameter(-73),
        'total_distal_weight': Parameter(0.008), #0.003 at 1 hz
        'distal_weight':Parameter(0.00001), #0.01
        'oblique_weight':Parameter(0.00001),
        'basal_weight':Parameter(0.00001),
        'syn_type':'additive',
        'interneuron_frequency': Parameter(75.0),
        'interneuron_reps': Parameter(1),
        },
    'shunt':
        {
        'reversal':Parameter(-73),
        'shunt_pos': Parameter(1),#0
        'tau1':Parameter(0.5),
        'tau2':Parameter(5),
        'basal_shunt_delay':Parameter(1.5),
        'shunt_delay':Parameter(1.5), #-4
        'distal_shunt_pos':Parameter(0.9),
        'proximal_shunt_pos':Parameter(0.18),
        'basal_shunt_pos':Parameter(0.3),
        'distal_shunt_compartment':'a',
        'proximal_shunt_compartment':'a',
        'basal_shunt_compartment':'basal',
        'distal_shunt_weight':Parameter([0.05]),
        'proximal_shunt_weight':Parameter([0.05]),
        'basal_shunt_weight':Parameter([0.05]),
        },
   'STDP':
       {
       'delta_t': Parameter(4,'ms'), #-1
       'thresh' : Parameter(-30, 'mV'),
       'ca_thresh' : Parameter(0.5, 'mM'),
       'potentiation_factor': Parameter(0.00106),
       'depression_factor': Parameter(0.001),
       'w_max': Parameter(0.0001),
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
    'cell':
        {
        'gsca' : Parameter(1.5),
        'git2' : Parameter(0.005),
        'gbar_kca' : Parameter(2.5),
        },
    'plot':
        {
        'version':Parameter(1),
        'path': os.path.expanduser("/scratch/kwilmes/Project1")
        }
}
