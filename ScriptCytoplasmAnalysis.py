# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:33:46 2026

@author: Joseph Vermeil

UtilityFunctions.py - contains all kind of small functions used by CortExplore programs, 
to be imported with "import UtilityFunctions as ufun" and call with "ufun.my_function".
Joseph Vermeil, 2026

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

#### Imports

import os
import re
import time

import numpy as np
import pandas as pd
import skimage as skm
import seaborn as sns
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import shapely
from shapely.ops import polylabel
from shapely.plotting import plot_polygon, plot_points # , plot_line

from trackpy.motion import msd, imsd, emsd
from PIL import Image, ImageDraw
from scipy import signal # stats #, optimize, interpolate, 

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.UtilityFunctions as ufun
import Libs.ToolboxCytoplasmAnalysis as tbca

#### Settings

SCALE_20X = 0.461
SCALE_40X = 0.229