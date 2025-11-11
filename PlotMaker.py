# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:52:35 2022
@author: Joseph Vermeil

ArticlePlotMaker.py - state the graphic styles elements of CortExplore programs, 
to be imported with "import GraphicStyles as gs" and call content with 
"my_graphic_style_thingy".
Joseph Vermeil, 2022

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# %% 0. Imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import re
import sys
import time
import random
import numbers
import warnings
import itertools
import matplotlib

from cycler import cycler
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest

#### Local Imports

import UtilityFunctions as ufun


#### Potentially useful lines of code
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# cp.DirDataFigToday

#### Pandas
pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_columns')
pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')


####  Matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})



# %% 1. Settings

# %%% 1.1 Global constants

cm_in = 2.52

# %%% 1.2 Useful lists 

# %%%% 1.2.1 Marker Lists
my_default_marker_list = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']
markerList10 = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']

# %%%% 1.2.2 Color Lists

# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
my_default_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
my_default_color_cycle = cycler(color=my_default_color_list)
plt.rcParams['axes.prop_cycle'] = my_default_color_cycle

pairedPalette = sns.color_palette("tab20")
pairedPalette = pairedPalette.as_hex()

colorList10 = my_default_color_list
sns.color_palette(my_default_color_list)

bigPalette1 = sns.color_palette("tab20b")
bigPalette1_hex = bigPalette1.as_hex()
bigPalette2 = sns.color_palette("tab20c")
bigPalette2_hex = bigPalette2.as_hex()

customPalette_hex = []
for ii in range(2, -1, -1):
    customPalette_hex.append(bigPalette2_hex[4*0 + ii]) # blue
    customPalette_hex.append(bigPalette2_hex[4*1 + ii]) # orange
    customPalette_hex.append(bigPalette2_hex[4*2 + ii]) # green
    customPalette_hex.append(bigPalette1_hex[4*3 + ii]) # red
    customPalette_hex.append(bigPalette2_hex[4*3 + ii]) # purple
    customPalette_hex.append(bigPalette1_hex[4*2 + ii]) # yellow-brown
    customPalette_hex.append(bigPalette1_hex[4*4 + ii]) # pink
    customPalette_hex.append(bigPalette1_hex[4*0 + ii]) # navy    
    customPalette_hex.append(bigPalette1_hex[4*1 + ii]) # yellow-green
    customPalette_hex.append(bigPalette2_hex[4*4 + ii]) # gray
    
# customPalette = sns.color_palette(customPalette_hex)
colorList30 = customPalette_hex

customPalette_hex = []
for ii in range(3, -1, -1):
    customPalette_hex.append(bigPalette2_hex[4*0 + ii]) # blue
    customPalette_hex.append(bigPalette2_hex[4*1 + ii]) # orange
    customPalette_hex.append(bigPalette2_hex[4*2 + ii]) # green
    customPalette_hex.append(bigPalette1_hex[4*3 + ii]) # red
    customPalette_hex.append(bigPalette2_hex[4*3 + ii]) # purple
    customPalette_hex.append(bigPalette1_hex[4*2 + ii]) # yellow-brown
    customPalette_hex.append(bigPalette1_hex[4*4 + ii]) # pink
    customPalette_hex.append(bigPalette1_hex[4*0 + ii]) # navy    
    customPalette_hex.append(bigPalette1_hex[4*1 + ii]) # yellow-green
    customPalette_hex.append(bigPalette2_hex[4*4 + ii]) # gray
    
colorList40 = customPalette_hex

palette_Set2 = sns.color_palette("Set2")
cL_Set2 = palette_Set2.as_hex()

palette_Set1 = sns.color_palette("Set1")
cL_Set1 = palette_Set1.as_hex()[:-1]

cL_Set12 = cL_Set1 + cL_Set2
cL_Set21 = cL_Set2 + cL_Set1

# %%%% 1.2.3 Display color / marker lists

def colorTester(colorList = colorList40):
    N = len(colorList)
    Nx = min(N, 10)
    Ny = 1+(N//10)
    X = np.arange(1, Nx+1)
    Y = np.arange(10, 10*(Ny+1), 10)
    fig, ax = plt.subplots(1, 1, figsize = (0.75*Nx, 1.25*Ny))
    for j in range(Ny):
        for i in range(Nx):
            try:
                ax.plot([X[i]], [Y[j]], color = colorList[i+10*j], marker = 'o', 
                        ls = '', markersize = 30, markeredgecolor = 'k')
            except:
                pass
    ax.set_xticks([i for i in X])
    ax.set_yticks([j for j in Y])
    ax.set_xticklabels([i for i in X])
    ax.set_yticklabels([j for j in Y])
    ax.set_ylim([0, 10*(Ny+1)])
    # ax.set_xlabel(r'colorList')
    # ax.set_ylabel('markerList')
    plt.tight_layout()
    plt.show()
    
def markerTester(markerList = markerList10):
    N = len(markerList)
    Nx = min(N, 10)
    Ny = 1+(N//10)
    X = np.arange(1, Nx+1)
    Y = np.arange(10, 10*(Ny+1), 10)
    fig, ax = plt.subplots(1, 1, figsize = (0.75*Nx, 1.25*Ny))
    for j in range(Ny):
        for i in range(Nx):
            try:
                ax.plot([X[i]], [Y[j]], color = 'w', marker = markerList[i], 
                        ls = '', markersize = 30, markeredgecolor = 'k')
            except:
                pass
    ax.set_xticks([i for i in X])
    ax.set_yticks([j for j in Y])
    ax.set_xticklabels([i for i in X])
    ax.set_yticklabels([j for j in Y])
    ax.set_ylim([0, 10*(Ny+1)])
    # ax.set_xlabel(r'colorList')
    # ax.set_ylabel('markerList')
    plt.tight_layout()
    plt.show()
    
# colorTester(colorList = cL_Set12)
# markerTester(markerList = markerList10)


# %%%% 1.2.4 Console text styles

NORMAL  = '\033[1;0m'
RED  = '\033[0;31m' # red
GREEN = '\033[1;32m' # green
ORANGE  = '\033[0;33m' # orange
BLUE  = '\033[0;36m' # blue
CYAN  = '\033[1;36m' # cyan
YELLOW = '\033[1;33m' # yellow
PURPLE = '\033[1;35m' # purple
GREY = '\033[1;30m'
DARKGREEN = '\033[0;32m' # green
BRIGHTRED = '\033[1;31m' # red
BRIGHTORANGE = '\033[1;33m' # orange
DARKPURPLE = '\033[1;35m' # purple

def consoleTextTester_01():
    print(NORMAL + 'normal' + NORMAL)
    print(RED + 'nothing rhyme with red' + NORMAL)
    print(ORANGE + 'nothing rhyme with orange' + NORMAL)
    print(YELLOW + 'nothing rhyme with yellow' + NORMAL)
    print(GREEN + 'nothing rhyme with green' + NORMAL)
    print(CYAN + 'nothing rhyme with cyan' + NORMAL)
    print(BLUE + 'nothing rhyme with blue' + NORMAL)
    print(PURPLE + 'nothing rhyme with purple' + NORMAL)
    print('\n')
    
    print(NORMAL + 'normal' + NORMAL)
    print(GREY + 'nothing rhyme with grey' + NORMAL)
    print(DARKGREEN + 'nothing rhyme with dark green' + NORMAL)
    print(BRIGHTRED + 'nothing rhyme with bright red' + NORMAL)
    print(BRIGHTORANGE + 'nothing rhyme with bright orange' + NORMAL)
    print(DARKPURPLE + 'nothing rhyme with dark purple' + NORMAL)
    print('\n')

def consoleTextTester_02():
    print("\033[0;37;48m Normal text\n")
    print("\033[2;37;48m Underlined text\033[0;37;48m \n")
    print("\033[1;37;48m Bright Colour\033[0;37;48m \n")
    print("\033[3;37;48m Negative Colour\033[0;37;48m \n")
    print("\033[5;37;48m Negative Colour\033[0;37;48m\n")
    print("\033[1;37;40m \033[2;37:40m TextColour BlackBackground          TextColour GreyBackground                WhiteText ColouredBackground\033[0;37;40m\n")
    print("\033[1;30;40m Dark Gray      \033[0m 1;30;40m            \033[0;30;47m Black      \033[0m 0;30;47m               \033[0;37;41m Black      \033[0m 0;37;41m")
    print("\033[1;31;40m Bright Red     \033[0m 1;31;40m            \033[0;31;47m Red        \033[0m 0;31;47m               \033[0;37;42m Black      \033[0m 0;37;42m")
    print("\033[1;32;40m Bright Green   \033[0m 1;32;40m            \033[0;32;47m Green      \033[0m 0;32;47m               \033[0;37;43m Black      \033[0m 0;37;43m")
    print("\033[1;33;48m Yellow         \033[0m 1;33;48m            \033[0;33;47m Brown      \033[0m 0;33;47m               \033[0;37;44m Black      \033[0m 0;37;44m")
    print("\033[1;34;40m Bright Blue    \033[0m 1;34;40m            \033[0;34;47m Blue       \033[0m 0;34;47m               \033[0;37;45m Black      \033[0m 0;37;45m")
    print("\033[1;35;40m Bright Magenta \033[0m 1;35;40m            \033[0;35;47m Magenta    \033[0m 0;35;47m               \033[0;37;46m Black      \033[0m 0;37;46m")
    print("\033[1;36;40m Bright Cyan    \033[0m 1;36;40m            \033[0;36;47m Cyan       \033[0m 0;36;47m               \033[0;37;47m Black      \033[0m 0;37;47m")
    print("\033[1;37;40m White          \033[0m 1;37;40m            \033[0;37;40m Light Grey \033[0m 0;37;40m               \033[0;37;48m Black      \033[0m 0;37;48m")
    print("\n")
    
# consoleTextTester_01()
# consoleTextTester_02()
    
    
    
# %%% 1.3 Set default graphic options

def setGraphicOptions(mode = 'screen', palette = 'Set2', colorList = cL_Set21):
    if mode == 'screen':
        SMALLER_SIZE = 11
        SMALL_SIZE = 13
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 20
    elif mode == 'print':
        SMALLER_SIZE = 8
        SMALL_SIZE = 10
        MEDIUM_SIZE = 11
        BIGGER_SIZE = 12
        
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    sns.set_palette(sns.color_palette(palette))
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=colorList) 



# %%% 1.4 Declare Dictionnaries

# %%%% 1.4.1 Rename dict 

# renameDict = {# Variables
#                'SurroundingThickness': 'Median Thickness (nm)',
#                'surroundingThickness': 'Median Thickness (nm)',
#                'ctFieldThickness': 'Median Thickness (nm)',
#                'ctFieldFluctuAmpli' : 'Thickness Fluctuations; $D_9$-$D_1$ (nm)',
#                'EChadwick': 'E Chadwick (Pa)',
#                'medianThickness': 'Median Thickness (nm)',               
#                'fluctuAmpli': 'Fluctuations Amplitude (nm)',               
#                'meanFluoPeakAmplitude' : 'Fluo Intensity (a.u.)', 
#                'fit_K' : 'Tangeantial Modulus (Pa)',
#                'bestH0' : 'Fitted $H_0$ (nm)',
#                'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
#                'E_f_<_400_kPa' : 'Elastic modulus (kPa)\nfor F < 400pN',
#                # Drugs
#                'none':'Control',
#                'dmso':'DMSO',
#                'blebbistatin':'Blebbi',
#                'latrunculinA':'LatA',
#                'Y27':'Y27',
#                }

# renameDict.update({'none & 0.0' : 'No drug',
#               'Y27 & 10.0' : 'Y27 10 µM', 
#               'Y27 & 50.0' : 'Y27 50 µM', 
#               'Y27 & 100.0' : 'Y27 100 µM', 
#               'dmso & 0.0' : 'DMSO',
#               'blebbistatin & 10.0' : 'Blebbi 10 µM', 
#               'blebbistatin & 50.0' : 'Blebbi 50 µM', 
#               'blebbistatin & 100.0' : 'Blebbi 100 µM',
#               'blebbistatin & 250.0' : 'Blebbi 250 µM', 
#               'LIMKi & 10.0' : 'LIMKi3 10 µM', 
#               'LIMKi & 20.0' : 'LIMKi3 20 µM', 
#               'latrunculinA & 0.1' : 'LatA 0.1 µM', 
#               'latrunculinA & 0.5' : 'LatA 0.5 µM', 
#               'latrunculinA & 2.5' : 'LatA 2.5 µM', 
#               'ck666 & 50.0' : 'CK666 50 µM', 
#               'ck666 & 100.0' : 'CK666 100 µM', 
#               'calyculinA & 0.25' : 'CalA 0.25µM',
#               'calyculinA & 0.5' : 'CalA 0.5µM',
#               'calyculinA & 1.0' : 'CalA 1.0µM',
#               'calyculinA & 2.0' : 'CalA 2.0µM',
#               })


# %%%% 1.4.2 Style dict 

# styleDict =  {# Drugs
#                'none':{'color': plt.cm.Greys(0.2),'marker':'o'},
#                'none & 0.0':{'color': plt.cm.Greys(0.2),'marker':'o'},
#                #
#                'dmso':{'color': plt.cm.Greys(0.5),'marker':'o'},
#                'dmso & 0.0':{'color': plt.cm.Greys(0.5),'marker':'o'},
#                #
#                'blebbistatin':{'color': plt.cm.RdPu(0.5),'marker':'o'},
#                'blebbistatin & 10.0':{'color':plt.cm.RdPu(0.4), 'marker':'o'},
#                'blebbistatin & 50.0':{'color': plt.cm.RdPu(0.65),'marker':'o'},
#                'blebbistatin & 100.0':{'color': plt.cm.RdPu(0.9),'marker':'o'},
#                'blebbistatin & 250.0':{'color': plt.cm.RdPu(1.0),'marker':'o'},
#                #
#                'PNB & 50.0':{'color': colorList40[25],'marker':'o'},
#                'PNB & 250.0':{'color': colorList40[35],'marker':'o'},
#                #
#                'latrunculinA':{'color': plt.cm.RdYlBu_r(0.75),'marker':'o'},
#                'latrunculinA & 0.1':{'color': plt.cm.RdYlBu_r(0.25),'marker':'s'},
#                'latrunculinA & 0.5':{'color': plt.cm.RdYlBu_r(0.75),'marker':'s'},
#                'latrunculinA & 2.5':{'color': plt.cm.RdYlBu_r(0.95),'marker':'s'},
#                #
#                'calyculinA':{'color': plt.cm.viridis_r(0.55),'marker':'o'},
#                'calyculinA & 0.25':{'color': plt.cm.viridis_r(0.05),'marker':'o'},
#                'calyculinA & 0.5':{'color': plt.cm.viridis_r(0.15),'marker':'o'},
#                'calyculinA & 1.0':{'color': plt.cm.viridis_r(0.4),'marker':'o'},
#                'calyculinA & 2.0':{'color': plt.cm.viridis_r(0.75),'marker':'o'},
#                #
#                'Y27':{'color': plt.cm.GnBu(0.3),'marker':'o'},
#                'Y27 & 10.0':{'color': plt.cm.GnBu(0.4),'marker':'^'},
#                'Y27 & 50.0':{'color': plt.cm.GnBu(0.7),'marker':'^'},
#                'Y27 & 100.0':{'color': plt.cm.GnBu(0.9),'marker':'^'},
#                #
#                'LIMKi':{'color': plt.cm.OrRd(0.5),'marker':'o'},
#                'LIMKi & 10.0':{'color': plt.cm.OrRd(0.45),'marker':'o'},
#                'LIMKi & 20.0':{'color': plt.cm.OrRd(0.9),'marker':'o'},
#                #
#                'JLY':{'color': colorList40[23],'marker':'o'},
#                'JLY & 8-5-10':{'color': colorList40[23],'marker':'o'},
#                #
#                'ck666':{'color': colorList40[25],'marker':'o'},
#                'ck666 & 50.0':{'color': colorList40[15],'marker':'o'},
#                'ck666 & 100.0':{'color': colorList40[38],'marker':'o'},
               
#                # Cell types
#                '3T3':{'color': colorList40[30],'marker':'o'},
#                'HoxB8-Macro':{'color': colorList40[32],'marker':'o'},
#                'DC':{'color': colorList40[33],'marker':'o'},
#                # Cell subtypes
#                'aSFL':{'color': colorList40[12],'marker':'o'},
#                'Atcc-2023':{'color': colorList40[10],'marker':'o'},
#                'optoRhoA':{'color': colorList40[13],'marker':'o'},
               
#                # Drugs + cell types
#                'aSFL-LG+++ & dmso':{'color': colorList40[19],'marker':'o'},
#                'aSFL-LG+++ & blebbistatin':{'color': colorList40[32],'marker':'o'},
#                'Atcc-2023 & dmso':{'color': colorList40[19],'marker':'o'},
#                'Atcc-2023 & blebbistatin':{'color': colorList40[22],'marker':'o'},
#                #
#                'Atcc-2023 & none':{'color': colorList40[0],'marker':'o'},
#                'Atcc-2023 & none & 0.0':{'color': colorList40[0],'marker':'o'},
#                'Atcc-2023 & Y27': {'color': colorList40[17],'marker':'o'},
#                'Atcc-2023 & Y27 & 10.0': {'color': colorList40[17],'marker':'o'},
#                #
#                'optoRhoA & none':{'color': colorList40[13],'marker':'o'},
#                'optoRhoA & none & 0.0':{'color': colorList40[13],'marker':'o'},
#                'optoRhoA & Y27': {'color': colorList40[27],'marker':'o'},
#                'optoRhoA & Y27 & 10.0': {'color': colorList40[27],'marker':'o'},
#                #
#                'Atcc-2023 & dmso':{'color': colorList40[9],'marker':'o'},
#                'Atcc-2023 & dmso & 0.0':{'color': colorList40[9],'marker':'o'},
#                'Atcc-2023 & blebbistatin':{'color': colorList40[12],'marker':'o'},
#                'Atcc-2023 & blebbistatin & 10.0':{'color': colorList40[2],'marker':'o'},
#                'Atcc-2023 & blebbistatin & 50.0':{'color': colorList40[12],'marker':'o'},
#                #
#                'optoRhoA & dmso':{'color': colorList40[29],'marker':'o'},
#                'optoRhoA & dmso & 0.0':{'color': colorList40[29],'marker':'o'},
#                'optoRhoA & blebbistatin':{'color': colorList40[32],'marker':'o'},
#                'optoRhoA & blebbistatin & 10.0':{'color': colorList40[22],'marker':'o'},
#                'optoRhoA & blebbistatin & 50.0':{'color': colorList40[32],'marker':'o'},
#                }


# %% 2. Data subfunctions

def filterDf(df, F):
    F = np.array(F)
    totalF = np.all(F, axis = 0)
    df_f = df[totalF]
    return(df_f)


def makeBoxPairs(O):
    return(list(itertools.combinations(O, 2)))


def makeCompositeCol(df, cols=[]):
    N = len(cols)
    if N > 1:
        newColName = ''
        for i in range(N):
            newColName += cols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        df[newColName] = ''
        for i in range(N):
            df[newColName] += df[cols[i]].astype(str)
            df[newColName] = df[newColName].apply(lambda x : x + ' & ')
        df[newColName] = df[newColName].apply(lambda x : x[:-3])
    else:
        newColName = cols[0]
    return(df, newColName)


def dataGroup(df, groupCol = 'cellID', idCols = [], numCols = [], aggFun = 'mean'):
    agg_dict = {'date':'first',
                'cellName':'first',
                'cellID':'first',	
                'manipID':'first',	
                'compNum':'count',
                }
    for col in idCols:
        agg_dict[col] = 'first'
    for col in numCols:
        agg_dict[col] = aggFun
    
    all_cols = list(agg_dict.keys())
    if not groupCol in all_cols:
        all_cols.append(groupCol)
        
    group = df[all_cols].groupby(groupCol)
    df_agg = group.agg(agg_dict)
    return(df_agg)


def dataGroup_weightedAverage(df, groupCol = 'cellID', idCols = [], 
                              valCol = '', weightCol = '', weight_method = 'ciw^2'):   
    idCols = ['date', 'cellName', 'cellID', 'manipID'] + idCols
    
    wAvgCol = valCol + '_wAvg'
    wVarCol = valCol + '_wVar'
    wStdCol = valCol + '_wStd'
    wSteCol = valCol + '_wSte'
    
    # 1. Compute the weights if necessary
    if weight_method == 'ciw^1':
        ciwCol = weightCol
        weightCol = valCol + '_weight'
        df[weightCol] = (df[valCol]/df[ciwCol])
    elif weight_method == 'ciw^2':
        ciwCol = weightCol
        weightCol = valCol + '_weight'
        df[weightCol] = (df[valCol]/df[ciwCol])**2
    
    df = df.dropna(subset = [weightCol])
    
    # 2. Group and average
    groupColVals = df[groupCol].unique()
    
    d_agg = {k:'first' for k in idCols}
    d_agg.update({'compNum':'count'})

    # In the following lines, the weighted average and weighted variance are computed
    # using new columns as intermediates in the computation.
    #
    # Col 'A' = K x Weight --- Used to compute the weighted average.
    # 'K_wAvg' = sum('A')/sum('weight') in each category (group by condCol and 'fit_center')
    #
    # Col 'B' = (K - K_wAvg)**2 --- Used to compute the weighted variance.
    # Col 'C' =  B * Weight     --- Used to compute the weighted variance.
    # 'K_wVar' = sum('C')/sum('weight') in each category (group by condCol and 'fit_center')
    
    # Compute the weighted mean
    df['A'] = df[valCol] * df[weightCol]
    grouped1 = df.groupby(by=[groupCol])
    d_agg.update({'A': ['count', 'sum'], weightCol: 'sum'})
    data_agg = grouped1.agg(d_agg).reset_index()
    data_agg.columns = ufun.flattenPandasIndex(data_agg.columns)
    data_agg[wAvgCol] = data_agg['A_sum']/data_agg[weightCol + '_sum']
    data_agg = data_agg.rename(columns = {'A_count' : 'count_wAvg'})
    data_agg = data_agg.rename(columns = {'compNum_count' : 'compNum'})
    
    # Compute the weighted std
    df['B'] = df[valCol]
    for co in groupColVals:
        weighted_avg_val = data_agg.loc[(data_agg[groupCol] == co), wAvgCol].values[0]
        index_loc = (df[groupCol] == co)
        col_loc = 'B'
        
        df.loc[index_loc, col_loc] = df.loc[index_loc, valCol] - weighted_avg_val
        df.loc[index_loc, col_loc] = df.loc[index_loc, col_loc] ** 2
            
    df['C'] = df['B'] * df[weightCol]
    grouped2 = df.groupby(by=[groupCol])
    data_agg2 = grouped2.agg({'C': 'sum', weightCol: 'sum'}).reset_index()
    data_agg2[wVarCol] = data_agg2['C']/data_agg2[weightCol]
    data_agg2[wStdCol] = data_agg2[wVarCol]**0.5
    
    # Combine all in data_agg
    data_agg[wVarCol] = data_agg2[wVarCol]
    data_agg[wStdCol] = data_agg2[wStdCol]
    data_agg[wSteCol] = data_agg[wStdCol] / data_agg['count_wAvg']**0.5
    
    # data_agg = data_agg.drop(columns = ['A_sum', weightCol + '_sum'])
    data_agg = data_agg.drop(columns = ['A_sum'])
    
    data_agg = data_agg.drop(columns = [groupCol + '_first'])
    data_agg = data_agg.rename(columns = {k+'_first':k for k in idCols})

    return(data_agg)


def makeCountDf(df, condition):   
    cols_count_df = ['compNum', 'cellID', 'manipID', 'date', condition]
    count_df = df[cols_count_df]
    groupByCell = count_df.groupby('cellID')
    d_agg = {'compNum':'count', condition:'first', 'date':'first', 'manipID':'first'}
    df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

    groupByCond = df_CountByCell.reset_index().groupby(condition)
    d_agg = {'cellID': 'count', 'compCount': 'sum', 
             'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
    d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
    df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
    
    return(df_CountByCond, df_CountByCell)


def pval2text(p, n_digits = 2, space=True):
    text='p-val'
    if space:
        if p >= 10**(-n_digits):
            text += f' = {p:.{n_digits}f}'
        else:
            text += f' < {10**(-n_digits):.{n_digits}f}'
    else:
        if p >= 10**(-n_digits):
            text += f'={p:.{n_digits}f}'
        else:
            text += f'<{10**(-n_digits):.{n_digits}f}'
    return(text)

# %% 3. Graphic subfunctions

def getSnsPalette(conditions, styleDict):
    colors = []
    try:
        for co in conditions:
            coStyle = styleDict[co]
            if 'color' in coStyle.keys():
                colors.append(coStyle['color'])
            else:
                colors.append('')
        palette = sns.color_palette(colors)
    except:
        palette = sns.color_palette(colorList10)
    return(palette)

def getStyleLists(conditions, styleDict):
    colors = []
    markers = []
    try:
        for co in conditions:
            coStyle = styleDict[co]
            colors.append(coStyle['color'])
            markers.append(coStyle['marker'])
    except:
        N = len(conditions)
        colors = colorList10
        markers = ['o'] * N
        
    return(colors, markers)

def renameAxes(axes, rD, format_xticks = True, rotation = 0):
    try:
        N = len(axes)
    except:
        axes = [axes]
        N = 1
    for i in range(N):
        # set xlabel
        xlabel = axes[i].get_xlabel()
        newXlabel = rD.get(xlabel, xlabel)
        axes[i].set_xlabel(newXlabel)
        # set ylabel
        ylabel = axes[i].get_ylabel()
        newYlabel = rD.get(ylabel, ylabel)
        axes[i].set_ylabel(newYlabel)
        # set title
        axtitle = axes[i].get_title()
        newaxtitle = rD.get(axtitle, axtitle)
        axes[i].set_title(newaxtitle)
        
        if format_xticks:
            # set xticks
            xticksTextObject = axes[i].get_xticklabels()
            xticksList = [xticksTextObject[j].get_text() for j in range(len(xticksTextObject))]
            test_hasXLabels = (len(''.join(xticksList)) > 0)
            if test_hasXLabels:
                newXticksList = [rD.get(k, k) for k in xticksList]
                axes[i].set_xticklabels(newXticksList, rotation = rotation)
                
                
def renameLegend(axes, rD, loc='best', ncols=1, fontsize=6, hlen=2):
    axes = ufun.toList(axes)
    N = len(axes)
    for i in range(N):
        ax = axes[i]
        L = ax.legend(loc = loc, ncols=ncols, fontsize=fontsize, handlelength=hlen)
        Ltext = L.get_texts()
        M = len(Ltext)
        for j in range(M):
            T = Ltext[j].get_text()
            for s in rD.keys():
                if re.search(s, T):
                    Ltext[j].set_text(re.sub(s, rD[s], T))
                    Ltext[j].set_fontsize(fontsize)
               
                
               
def addStat_lib(ax, box_pairs, test = 'Mann-Whitney', verbose = False, **plotting_parameters):
    #### STATS
    listTests = ['t-test_ind', 't-test_welch', 't-test_paired', 
                 'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls', 
                 'Levene', 'Wilcoxon', 'Kruskal', 'Brunner-Munzel']
    if test in listTests:
        annotator = Annotator(ax, box_pairs, **plotting_parameters)
        annotator.configure(test=test, verbose=verbose, fontsize = 11,
                            line_height = 0.01, line_offset = 1, line_offset_to_group = 1)
        annotator.apply_and_annotate() 
        # , loc = 'outside', line_offset = -1, line_offset_to_group = -1
    else:
        print(BRIGHTORANGE + 'Dear Madam, dear Sir, i am the eternal god and i command that you define this stat test cause it is not in the list !' + NORMAL)
    return(ax)


# %% 4. Plot Functions

def D1Plot(data, fig = None, ax = None, condition='', parameter='',
           co_order=[], boxplot=1, figSizeFactor = 1, markersizeFactor = 1,
           stats=True, statMethod='Mann-Whitney', box_pairs=[], statVerbose = False,
           showMean = False, edgecolor='k'):
    
    #### Init
    co_values = data[condition].unique()
    Nco = len(co_values)
    if len(co_order) == 0:
        co_order = np.sort(co_values)
        
    if ax == None:
        figHeight = 5
        figWidth = 5*Nco*figSizeFactor
        fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
    markersize = 5 * markersizeFactor
    linewidth = 0.75*markersizeFactor
    
        
        
    palette = getSnsPalette(co_order, styleDict)
    
    #### Swarmplot
    swarmplot_parameters = {'data':    data,
                            'x':       condition,
                            'y':       parameter,
                            'order':   co_order,
                            'palette': palette,
                            'size'    : markersize, 
                            'edgecolor'    : edgecolor, 
                            'linewidth'    : linewidth
                            }
    if edgecolor != 'None':
        swarmplot_parameters['linewidth']=linewidth
    
    sns.swarmplot(ax=ax, **swarmplot_parameters)

    #### Stats    
    if stats:
        if len(box_pairs) == 0:
            box_pairs = makeBoxPairs(co_order)
        addStat_lib(ax, box_pairs, test = statMethod, verbose = statVerbose, **swarmplot_parameters)

    
    #### Boxplot
    if boxplot>0:
        boxplot_parameters = {'data':    data,
                                'x':       condition,
                                'y':       parameter,
                                'order':   co_order,
                                'width' : 0.5,
                                'showfliers': False,
                                }
        if boxplot==1:
            boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                    # boxprops={"color": color, "linewidth": 0.5},
                                    whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                    capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
        
        elif boxplot==2:
            boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                    # boxprops={"color": color, "linewidth": 0.5},
                                    whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                    capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2})
        
        
        elif boxplot == 3:
            boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 4},
                                boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1.5, 'alpha' : 0.75, 'zorder' : 4},
                                # boxprops={"color": color, "linewidth": 0.5},
                                whiskerprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.75, 'zorder' : 4},
                                capprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.75, 'zorder' : 4})

            
        if showMean:
            boxplot_parameters.update(meanline='True', showmeans='True',
                                      meanprops={"color": 'darkblue', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},)
            
        sns.boxplot(ax=ax, **boxplot_parameters)
        
    return(fig, ax)



def D1Plot_violin(data, fig = None, ax = None, condition='', parameter='',
           co_order=[], figSizeFactor = 1, 
           stats=True, statMethod='Mann-Whitney', box_pairs=[], statVerbose = False):
    
    #### Init
    co_values = data[condition].unique()
    Nco = len(co_values)
    if len(co_order) == 0:
        co_order = np.sort(co_values)
        
    if ax == None:
        figHeight = 5
        figWidth = 5*Nco*figSizeFactor
        fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))    
        
        
    palette = getSnsPalette(co_order, styleDict)
    
    #### Swarmplot
    violinplot_parameters = {'data':    data,
                            'x':       condition,
                            'y':       parameter,
                            'order':   co_order,
                            'palette': palette,
                            }
    
    sns.violinplot(ax=ax, **violinplot_parameters)

    #### Stats    
    if stats:
        if len(box_pairs) == 0:
            box_pairs = makeBoxPairs(co_order)
        addStat_lib(ax, box_pairs, test = statMethod, verbose = statVerbose, **violinplot_parameters)

    return(fig, ax)


def D2Plot_wFit(data, fig = None, ax = None, 
                XCol='', YCol='', condition='', co_order = [],
                modelFit=False, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1):
    
    #### Init
    co_values = data[condition].unique()
    print(co_values)
    Nco = len(co_values)
    if len(co_order) == 0:
        co_order = np.sort(co_values)
        
    if ax == None:
        figHeight = 5
        figWidth = 5*Nco*figSizeFactor
        fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
    markersize = 5 * markersizeFactor
        
    colors, markers = getStyleLists(co_order, styleDict)
    
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize = (8*figSizeFactor,5))
    else:
        pass
    
    markersize = 5 * markersizeFactor
    
    #### Get fitting function
    if robust == False:
        my_fitting_fun = ufun.fitLine
    else:
        my_fitting_fun = ufun.fitLineHuber
    
    
    #### Get data for fit
    for i in range(Nco):
        cond = co_order[i]
        c = colors[i]
        m = markers[i]
        Xraw = data[data[condition] == cond][XCol].values
        Yraw = data[data[condition] == cond][YCol].values
        Mraw = data[data[condition] == cond]['manipID'].values
        XYraw = np.array([Xraw,Yraw]).T
        XY = XYraw[~np.isnan(XYraw).any(axis=1), :]
        X, Y = XY[:,0], XY[:,1]
        M = Mraw[~np.isnan(XYraw).any(axis=1)]
        if len(X) == 0:
            ax.plot([], [])
            if modelFit:
                ax.plot([], [])
                
        elif len(X) > 0:
            eqnText = ''

            if modelFit:
                print('Fitting condition ' + cond + ' with model ' + modelType)
                if modelType == 'y=ax+b':
                    params, results = my_fitting_fun(X, Y) 
                    # Y=a*X+b ; params[0] = b,  params[1] = a
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    eqnText += " ; Y = {:.1f} X + {:.1f}".format(params[1], params[0])
                    eqnText += "\np-val = {:.3f}".format(pval)
                    print("Y = {:.5} X + {:.5}".format(params[1], params[0]))
                    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = params[1]*fitX + params[0]
                    ax.plot(fitX, fitY, '--', lw = '2', 
                            color = c, zorder = 4)

                elif modelType == 'y=A*exp(kx)':
                    params, results = my_fitting_fun(X, np.log(Y)) 
                    # Y=a*X+b ; params[0] = b,  params[1] = a
                    pval = results.pvalues[1] # pvalue on the param 'k'
                    eqnText += " ; Y = {:.1f}*exp({:.1f}*X)".format(params[0], params[1])
                    eqnText += "\np-val = {:.3f}".format(pval)
                    print("Y = {:.5}*exp({:.5}*X)".format(np.exp(params[0]), params[1]))
                    print("p-value on the 'k' coefficient: {:.4e}".format(pval))
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = np.exp(params[0])*np.exp(params[1]*fitX)
                    ax.plot(fitX, fitY, '--', lw = '2', 
                            color = c, zorder = 4)
                    
                elif modelType == 'y=k*x^a':
                    posValues = ((X > 0) & (Y > 0))
                    X, Y = X[posValues], Y[posValues]
                    params, results = my_fitting_fun(np.log(X), np.log(Y)) 
                    # Y=a*X+b ; params[0] = b,  params[1] = a
                    k = np.exp(params[0])
                    a = params[1]
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    eqnText += " ; Y = {:.1e} * X^{:.1f}".format(k, a)
                    eqnText += "\np-val = {:.3f}".format(pval)
                    print("Y = {:.4e} * X^{:.4f}".format(k, a))
                    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = k * fitX**a
                    ax.plot(fitX, fitY, '--', lw = '2', 
                            color = c, zorder = 4)
                
                print('Number of values : {:.0f}'.format(len(Y)))
                print('\n')
            
            labelText = cond
            if writeEqn:
                labelText += eqnText
            if robust:
                labelText += ' (R)'

            ax.plot(X, Y, 
                    color = c, ls = '', 
                    marker = m, markersize = markersize, 
                    # markeredgecolor='k', markeredgewidth = 1, 
                    label = labelText)
            
    ax.set_xlabel(XCol)
    ax.set_ylabel(YCol)
    ax.legend()
    
    return(fig, ax)


