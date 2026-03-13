#!/usr/bin/env python


"""config for neuronmodel.py"""

# _title_     : config_neuronmodel.py
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


params = {
    'visual': 'figure',
    'results_file': '',
    'Neuron':
        {
        # morphology
        'a_diam':Parameter(2),
        's_diam':Parameter(18.5),
        'd_diam':Parameter(2),
        'd_length':Parameter(300),              
        'apical_length':Parameter(500),              
        'n_seg':Parameter(151),              
        # passive parameters
        'R_m':Parameter(40000),
        'R_a':Parameter(150),
        'C_m':Parameter(0.75),              
        'E_leak':Parameter(-70),
        'V_rest':Parameter(-70),
        # active conductances
        'E_Na':Parameter(60),
        'E_K':Parameter(-80),
        'E_Ca':Parameter(140),        
        'g_Na':Parameter(0.009),
        'g_K':Parameter(0.01),
        'g_KA':Parameter(0.029),
        'slope_KA':Parameter(5),

        # calcium
        'gsca': Parameter(1.5,'pS/um^2'), 
        'git2': Parameter(0.005,'S/cm^2'), 
        'g_KCa': Parameter(2.5,'pS/um^2'), 
        'ifca': True,
           
        # ais
        'g_Na_ais':Parameter(0.3),
        'g_Na_ais_shifted':Parameter(0.3),
        'ifshift':True,
        'dend_vshift':Parameter(-5)
        },
    'sim':
        {
        'duration' : Parameter(1,'second'),
        },
}






