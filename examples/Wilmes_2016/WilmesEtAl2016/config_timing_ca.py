#!/usr/bin/env python


"""config-file for Figure 4B"""

# _title_     : config_timing_ca.py
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

from NeuroTools.parameters import Parameter, ParameterSet, ParameterSpace, ParameterRange



#param_ranges = {
#   'Input':
#       {
#       'freq': ParameterArray(np.arange(10,20,10)),
#       },
#   'Synapse':
#       {
#        'pos': ParameterArray([3]), #1,3
#        'shunt_pos': ParameterArray([1]), #0,1,2
#        'shunt_weight':ParameterArray(np.arange(0.0,0.1,0.01)),
#        'shunt_delay':ParameterArray(np.arange(-15,10,0.1)),
#        'shunt_reversal':ParameterArray([-73])
#       }
#
#   }

params = {
    'condition':'ca',
    'Input':
        {
        'freq': Parameter(10),
        'wee': Parameter(0.02)
        },
    'Synapse':
        {
        'pos': Parameter(3),
        'distal_weight':Parameter(0.008),
        'oblique_weight':Parameter(0.00001),
        'basal_weight':Parameter(0.00001),
        'syn_type':'additive',
        'AP_DELAY':Parameter(0)
        },
    'shunt':
        {
        'delay_start':Parameter(-15),
        'delay_end':Parameter(10),
        'reversal':Parameter(-73),
        'shunt_pos': Parameter(1),#0
        'tau1':Parameter(0.5),
        'tau2':Parameter(5),
        'distal_shunt_pos':Parameter(0.9),
        'proximal_shunt_pos':Parameter(0.18),
        'basal_shunt_pos':Parameter(0.3),
        'distal_shunt_compartment':'a',
        'proximal_shunt_compartment':'a',
        'basal_shunt_compartment':'basal',
        },
    'STDP':
       {
       'delta_t': Parameter(-5,'ms')
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
