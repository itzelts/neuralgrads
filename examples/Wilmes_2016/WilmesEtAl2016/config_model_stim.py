#!/usr/bin/env python


"""config-file for model_stim.py, which creates Figures 1B-E"""

# _title_     : config_model_stim.py
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
sys.path.append("/home/wilmes/wilmes/scripts/project/model/lib/python2.6/site-packages/")

from NeuroTools.parameters import Parameter, ParameterSet, ParameterSpace, ParameterRange

#from snep.utils import Parameter, ParameterArray
#from snep.experiment import Experiment

"""
param_ranges = {
   'Input':
       {
       'freq': ParameterArray([10]),
       'wee':ParameterArray([0.008]),
       'wee_strong':ParameterArray([0.014]),
       },
   'Synapse':
       {
        'scen': ParameterArray([1,2,3,4]),
       },
   'cell':
       {
       'git2' : ParameterArray([0.005]),
       'gbar_kca' : ParameterArray([2.5]),
       }

   }
"""

params = {
    'visual': 'figure',
    'results_file': 'pairingscenarios',
    'Input':
        {
        'freq': Parameter(40),
        'wee':Parameter(0.008),
        'wee_strong':Parameter(0.014),
        },
    'Synapse':
        {
        'shunt_reversal':Parameter(-73),
        'pos': Parameter(3),
        'shunt_pos': Parameter(1),
        'shunt_tau':Parameter(5),
        'basal_shunt_delay':Parameter(-4),
        'shunt_delay':Parameter(-4), #-4
        'distal_weight':Parameter(0.00001), #0.01
        'oblique_weight':Parameter(0.00001),
        'basal_weight':Parameter(0.00001),
        'distal_shunt_weight':Parameter(0.01),
        'proximal_shunt_weight':Parameter(0.04),
        'basal_shunt_weight':Parameter(0.04),
        'syn_type':'additive'
        },
   'STDP':
       {
       'delta_t': Parameter(-5,'ms'),
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
