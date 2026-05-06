# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:10:16 2026

@author: Utilisateur
"""

# %% Import
import numpy as np


# %% Fitting functions

def doubleExpo(x, A, k1, B, k2):
    return(A*np.exp(-x/k1) + B*np.exp(-x/k2))

def powerLaw(x, A, k):
    return(A*(x**k))


# %% Constants per magnets

dict_magnet_MA = {
    'MyOne':{
        "F_popt_2exp": [
            6.695570103435238,
            104.27091113638627,
            0.12369116542822649,
            25350957.444718868
        ],
        "F_popt_pL": [
            23630.751398753007,
            -1.8872124731536342
        ],
    },
    'M270':{
        "F_popt_2exp": [
            141.59732873683043,
            72.16444694994189,
            12.460618814764926,
            272.91475019440617
        ],
        "F_popt_pL": [
            1482165.648896691,
            -2.166374314607223
        ],
    },
}


dict_magnet_JX = {
    'MyOne':{
        "F_popt_2exp": [
            5.99012502563366,
            100.84369389492127,
            0.11435249326662983,
            2101.719709808641
        ],
        "F_popt_pL": [
            39603.33040969049,
            -2.0162526263553215
        ],
    },
    
    'M270':{
        "F_popt_2exp": [
            119.21587409989525,
            65.9084742068066,
            16.320212861019403,
            214.41166616406406
        ],
        "F_popt_pL": [
            948854.7339381203,
            -2.1236746283466172
        ],
    },
}

dict_magnet_JN = {
    'MyOne':{
        "F_popt_2exp": [
            7.805139888548116,
            102.27510392873741,
            1.2304327867498293,
            270.4339259099587
        ],
        "F_popt_pL": [
            20963.176438241888,
            -1.785737243995788
        ],
    },
}


dict_magnet_JV01 = {
    'MyOne':{
        "F_popt_2exp": [
            0,
            0,
            0,
            0
        ],
        "F_popt_pL": [
            0,
            0
        ],
    },
}


dict_allMagnets = {
    'magnet_MA':dict_magnet_MA,
    'magnet_JX':dict_magnet_JX,
    'magnet_JN':dict_magnet_JN,
    'magnet_JV01':dict_magnet_JV01,
}


# %% Utility function

def getMagnet_D2F(magnet, beads, funcType):
    
    if funcType in ['power law', 'power-law', 'powerLaw', 'power_law', 'pl', 'pL', 'PL']:
        funcType = "F_popt_pL"
    elif funcType in ['2exp', '2Exp', '2_exp', '2_Exp', '2expo', '2Expo', 
                      'double expo', 'double Expo', 'doubleExpo', 'double_expo', 'double-expo',
                      'double exponential', 'double Exponential', 'doubleExponential', 
                      'double_exponential', 'double-exponential',]:
        funcType = "F_popt_2exp"
    
    D2F_func = dict_allMagnets[magnet][beads][funcType]
    
    return(D2F_func)
   