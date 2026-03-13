#!/usr/bin/env python


"""config-file for allornone.py, for Figure 2A"""

# _title_     : config_allornone.py
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


"""
param_ranges = {
   'Input':
       {
       'freq': ParameterArray([10])
       },
   'Synapse':
       {
        'shunt_weight':ParameterArray(np.arange(0,0.1,0.005)),
        'shunt_reversal':ParameterArray([-73]),
       }
   }
"""

params = {
    'visual': 'figure',
    'results_file': 'pairingscenarios',
    'Input':
        {
        'freq': Parameter(10)
        },
    'Synapse':
        {
        'pos': Parameter(3),
        'timing_sigma':Parameter(0),
        'distal_weight':Parameter(0.008),
        'oblique_weight':Parameter(0.001),
        'basal_weight':Parameter(0.005)
        },
    'shunt':
        {
        'shunt_weight':Parameter(0.05), 
        'shunt_sigma':Parameter(0.05),
        'shunt_pattern':'gauss',
        'shunt_number':Parameter(0),
        'shunt_delay':Parameter(2),
        'reversal':Parameter(-73),
        'tau1':Parameter(0.5),
        'tau2':Parameter(5),
        'shunt_compartment':'a',
        'exact_shunt_pos':Parameter(0.18),
        'distributed':Parameter(0)
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
