#!/usr/bin/env python


"""config-file for traces of Figure 4A"""

# _title_     : config_traces_ca.py
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
#       'freq': ParameterArray([10]),
#       'wee':ParameterArray([0.008]),
#       },
#   'Synapse':
#       {
#        'scen': ParameterArray([3]),
#        'shunt_pos':ParameterArray([0,1,2,3])
#       }
#
#   }

params = {
    'visual': 'figure',
    'results_file': 'pairingscenarios',
    'Input':
        {
        'freq': Parameter(10),
        'wee':Parameter(0.008), #0.002
        'wee_strong':Parameter(0.014) #0.002
        },
    'Synapse':
        {
        'scen': Parameter(3),
        'pos': Parameter(3),
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
        'shunt_pos': Parameter(0),
        'tau1':Parameter(0.5),
        'tau2':Parameter(5),
        'basal_shunt_delay':Parameter(1.5),
        'shunt_delay':Parameter(1.5), 
        'distal_shunt_weight':Parameter(0.05),
        'proximal_shunt_weight':Parameter(0.05),
        'basal_shunt_weight':Parameter(0.05),
        'distal_shunt_pos':Parameter(0.9),
        'proximal_shunt_pos':Parameter(0.18),
        'basal_shunt_pos':Parameter(0.3),
        'distal_shunt_compartment':'a',
        'proximal_shunt_compartment':'a',
        'basal_shunt_compartment':'basal',
        },
   'STDP':
       {
       'delta_t': Parameter(4,'ms'),
       'thresh' : Parameter(-30, 'mV'),
       'ca_thresh' : Parameter(0.5, 'mM')
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
