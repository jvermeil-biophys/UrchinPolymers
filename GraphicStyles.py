# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:52:35 2022
@author: Joseph Vermeil

GraphicStyles.py - state the graphic styles elements of CortExplore programs, 
to be imported with "import GraphicStyles as gs" and call content with 
"gs.my_graphic_style_thingy".
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
import matplotlib.pyplot as plt

import itertools
import matplotlib

from cycler import cycler



# %% 1. Useful styles

# %%% 1.1 Markers
my_default_marker_list = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']
markerList10 = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']

# %%% 1.2 Colors

# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
my_default_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
my_default_color_cycle = cycler(color=my_default_color_list)
plt.rcParams['axes.prop_cycle'] = my_default_color_cycle

pairedPalette = sns.color_palette("tab20")
pairedPalette = pairedPalette.as_hex()
pairedPalette

# clist = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a']
# sns.color_palette(clist)
colorList10 = my_default_color_list
sns.color_palette(my_default_color_list)


bigPalette1 = sns.color_palette("tab20b")
bigPalette1_hex = bigPalette1.as_hex()
bigPalette1

bigPalette2 = sns.color_palette("tab20c")
bigPalette2_hex = bigPalette2.as_hex()
bigPalette2

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

# %%% 1.3 Tester

def colorTester():
    colorList = colorList40
    markerList = markerList10
    N = len(colorList)
    M = len(markerList)
    X = np.arange(0, N)
    Y = np.arange(0, M)
    fig, ax = plt.subplots(1, 1, figsize = (0.4*N, 0.4*M))
    for i in range(N):
        for j in range(M):
            ax.plot([X[i]], [Y[-1-j]], color = colorList[i], marker = markerList[j], 
                    ls = '', markersize = 10, markeredgecolor = 'k')
    ax.set_xticks([i for i in range(N)])
    ax.set_yticks([j for j in range(M)])
    ax.set_xticklabels([i for i in range(N)])
    ax.set_yticklabels([j for j in range(M)])
    ax.set_xlabel(r'colorList')
    ax.set_ylabel('markerList')
    plt.tight_layout()
    plt.show()
    
# colorTester()

# %% 2. Global constants

# %%% 2.1 Console text styles

cm_in = 2.52

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

# %%% 2.2 Tester

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
    
    
    
# %% 3. Set default graphic options

SMALLER_SIZE = 8
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

def set_defense_options_jv(palette = 'Set2'):
    SMALLER_SIZE = 9
    SMALL_SIZE = 11
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    sns.set_palette(sns.color_palette(palette))


def set_default_options_jv():
    SMALLER_SIZE = 8
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 20
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    
def set_mediumText_options_jv():
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
    
def set_manuscript_options_jv(palette = 'Set2'):
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
    
def set_defense_options_jv(palette = 'Set2'):
    SMALLER_SIZE = 9
    SMALL_SIZE = 11
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    sns.set_palette(sns.color_palette(palette))

def set_smallText_options_jv():
    SMALLER_SIZE = 6
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
def set_smallText_options_aj():
    SMALLER_SIZE = 5
    SMALL_SIZE = 7
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    


def set_bigText_options_jv():
    SMALLER_SIZE = 14
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# set_default_options_jv()    

# %% Other color lists (cL)

palette_Set2 = sns.color_palette("Set2")
cL_Set2 = palette_Set2.as_hex()

palette_Set1 = sns.color_palette("Set1")
cL_Set1 = palette_Set1.as_hex()[:-1]

cL_Set12 = cL_Set1 + cL_Set2
cL_Set21 = cL_Set2 + cL_Set1

# X1, Y1 = np.arange(len(cL_Set1)), np.arange(len(cL_Set1))
# X2, Y2 = np.arange(len(cL_Set2)), np.arange(len(cL_Set2))-1
# fig, ax = plt.subplots(1, 1)
# ax.scatter(X1, Y1, c=cL_Set1, s=200)
# ax.scatter(X2, Y2, c=cL_Set2, s=200)
# plt.show()

# pal = sns.mpl_palette("Set2", 14)
# cL = pal.as_hex()

# X1, Y1 = np.arange(len(cL)), np.arange(len(cL))
# # X2, Y2 = np.arange(len(colorList_Set2)), np.arange(len(colorList_Set2))-1
# fig, ax = plt.subplots(1, 1)
# ax.scatter(X1, Y1, c=cL, s=200)
# # ax.scatter(X2, Y2, c=colorList_Set2, s=200)
# plt.show()

# %%

# c1 = plt.cm.GnBu(0.2)
# c2 = plt.cm.GnBu(0.4)
# c3 = plt.cm.GnBu(0.6)
# c4 = plt.cm.GnBu(0.8)

# cL = [c1, c2, c3, c4]

# X1, Y1 = np.arange(len(cL)), np.arange(len(cL))
# fig, ax = plt.subplots(1, 1)
# ax.scatter(X1, Y1, c=cL, s=200)

# plt.show()
