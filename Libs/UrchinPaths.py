# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 11:30:18 2026

@author: Utilisateur
"""

# %% 0. Imports

import os
import sys
import ctypes
from datetime import date

# %% 1. Paths

COMPUTERNAME = os.environ['COMPUTERNAME']

# 1.1 Init main directories

if COMPUTERNAME == 'PROCYON-PC': # Ordi Perso
    Path_AnalysisPulls = "C:\\Users\\josep\\Desktop\\Seafile\\DownloadedFromSeafile\\"
    Path_WorkingData = "E:\\WorkingData\\"
    Path_Fiji = "C:\\Users\\josep\\Desktop\\Fiji.app\\"
    Path_JAVA_HOME = "C:\\Users\\josep\\miniforge3\\envs\\pyimagej-env\\Library\\lib\\jvm"
    
elif COMPUTERNAME == 'MINC05': # IJM
    Path_AnalysisPulls = "E:\\AnalysisPulls\\"
    Path_WorkingData = "E:\\WorkingData\\"
    
    
elif COMPUTERNAME == 'DESKTOP-9J5NPMO': # LJP
    Path_AnalysisPulls = "C:\\Users\\Joseph\\Desktop\\AnalysisPulls\\"
    Path_IntraCellTracking = "C:\\Users\\Joseph\\Desktop\\IntraCellTracking\\"
    Path_WorkingData = "C:\\Users\\Joseph\\Desktop\\WorkingData\\"
    Path_Fiji = "C:\\Users\\Joseph\\Desktop\\Fiji_stable.app"
    Path_JAVA_HOME = "C:\\Users\\Joseph\\miniforge3\\envs\\spyder-env\\Library\\lib\\jvm"
    
Path_LeicaData = os.path.join(Path_WorkingData, "LeicaData")
Path_Nikon1Data = os.path.join(Path_WorkingData, "Nikon1Data_X1")
Path_Nikon2Data = os.path.join(Path_WorkingData, "Nikon2Data")
Path_Nikon3Data = os.path.join(Path_WorkingData, "Nikon3Data_W1")

