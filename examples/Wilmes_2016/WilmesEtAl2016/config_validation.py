#!/usr/bin/env python


"""config for critical_frequency.py"""

# _title_     : config_validation.py
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
#       'freq': ParameterArray(np.arange(10,100,10)),
#       #'freq': ParameterArray([10]),
#       #'mu_dendrite': ParameterArray([0.56]), #np.arange(0,1,0.1),'nA'), #0.56nA
#       #'scen': ParameterArray([0,1,2]), # 0,1,2 or just 2
#
#       },
#   'sim':
#       {
#       #'duration': ParameterArray((np.arange(50,100,50))),
#       #'Ca_prob': ParameterArray((np.arange(0,1.1,0.3)))
#       }
#
#   }

params = {
    'visual': 'figure',
    'results_file': 'pairingscenarios',
    'Input':
        {
        'freq': Parameter(10),
        'theta': Parameter(3,'ms'),
        'sigma': Parameter(0.3,'nA'),
        #'mu_dendrite': Parameter(0.75,'nA'),
        'stepduration': Parameter(250,'ms'),
        'stepnumber': Parameter(25),
        'stepsize': Parameter(0.02,'nA'), # 50pA
        },
    'Synapse':
        {
        'pos': Parameter(3),
        },
   'cell':
        {
        'gsca' : Parameter(1.5),
        'git2' : Parameter(0.005),
        'gbar_kca' : Parameter(2.5),
        'ra' : Parameter(150)
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
