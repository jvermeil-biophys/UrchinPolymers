# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:12:12 2025

@author: Utilisateur
"""

# %% 1. Imports

import matplotlib as mpl
import matplotlib.pyplot as plt

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

from glob import glob  # Used only for instructive purposes


# %% 2. 
