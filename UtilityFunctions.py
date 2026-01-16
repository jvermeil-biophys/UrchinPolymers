# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:21:02 2022
@authors: Joseph Vermeil, Anumita Jawahar

UtilityFunctions.py - contains all kind of small functions used by CortExplore programs, 
to be imported with "import UtilityFunctions as ufun" and call with "cp.my_function".
Joseph Vermeil, Anumita Jawahar, 2022

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

# %% (0) Imports and settings

# 1. Imports
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import os
import re
import time
import shutil
import random
import numbers
import matplotlib
import traceback

from scipy import interpolate
from scipy import signal
from scipy import odr

from PIL import Image
from PIL.TiffTags import TAGS
from ome_types import from_tiff

from skimage import io, filters, exposure, measure, transform, util, color
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment, least_squares
from matplotlib.gridspec import GridSpec
from datetime import date, datetime
from collections.abc import Collection
from copy import deepcopy

#### Local Imports

import GraphicStyles as gs


# %% (1) Utility functions


# %%% Image management - nomenclature and creating stacks
       

def renamePrefix(DirExt, currentCell, newPrefix):
    """
    Used for metamorph created files.
    Metamorph creates a new folder for each timelapse, within which all images contain a predefined 
    'prefix' and 'channel' name which can differ between microscopes. Eg.: 'w1TIRF_DIC' or 'w2TIRF_561'
    
    If you forget to create a new folder for a new timelapse, Metamorph automatically changes the prefix
    to distinguish between the old and new timelapse triplets. This can get annoying when it comes to processing 
    many cells.
    
    This function allows you to rename the prefix of all individual triplets in a specific folder. 
    
    """
    
    path = os.path.join(DirExt, currentCell)
    allImages = os.listdir(path)
    for i in allImages:
        if i.endswith('.TIF'):
            split = i.split('_')
            if split[0] != newPrefix:
                split[0] = newPrefix
                newName = '_'.join(split)
                try:
                    os.rename(os.path.join(path,i), os.path.join(path, newName))
                except:
                    print(currentCell)
                    print(gs.ORANGE + "Error! There may be other files with the new prefix you can trying to incorporate" + gs.NORMAL)
        
        if i.endswith('.nd'):
            newName = newPrefix+'.nd'
            try:
                os.rename(os.path.join(path,i), os.path.join(path, newName))
            except:
                print(currentCell)
                print(gs.YELLOW + "Error! There may be other .nd files with the new prefix you can trying to incorporate" + gs.NORMAL)
            
# %%% Data management

def removeColumnsDuplicate(df):
    cols = df.columns.values
    for c in cols:
        if c.endswith('_x'):
            df = df.rename(columns={c: c[:-2]})
        elif c.endswith('_y'):
            df = df.drop(columns=[c])
    return(df)



def findInfosInFileName(f, infoType):
    """
    Return a given type of info from a file name.
    Inputs : f (str), the file name.
             infoType (str), the type of info wanted.
             
             infoType can be equal to : 
                 
             * 'M', 'P', 'C' -> will return the number of manip (M), well (P), or cell (C) in a cellID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'C', the function will return 8.
             
             * 'manipID'     -> will return the full manip ID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'manipID', the function will return '21-01-18_M2'.
             
             * 'cellID'     -> will return the full cell ID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'cellID', the function will return '21-01-18_M2_P1_C8'.
             
             * 'substrate'  -> will return the string describing the disc used for cell adhesion.
             ex : if f = '21-01-18_M2_P1_C8_disc15um.tif' and infoType = 'substrate', the function will return 'disc15um'.
             
             * 'ordinalValue'  -> will return a value that can be used to order the cells. It is equal to M*1e6 + P*1e3 + C
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'ordinalValue', the function will return "2'001'008".
    """
    infoString = ''
    try:
        if infoType in ['M', 'P', 'C', 'cellName']:
            templateStr = r'M[0-9]_P[0-9]_C[0-9\-]+'
            s = re.search(templateStr, f)
            if s:
                iStart, iStop = s.span()
                foundStr = f[iStart:iStop]
                if infoType == 'cellName':
                    infoString = foundStr
                elif infoType == 'M':
                    infoString = foundStr.split('_')[0][1:]
                elif infoType == 'P':
                    infoString = foundStr.split('_')[1][1:]
                elif infoType == 'C':
                    infoString = foundStr.split('_')[2][1:]
                
        if infoType in ['M', 'P', 'C']:
            acceptedChar = [str(i) for i in range(10)] + ['-']
            string = '_' + infoType
            iStart = re.search(string, f).end()
            i = iStart
            infoString = '' + f[i]
            while f[i+1] in acceptedChar and i < len(f)-1:
                i += 1
                infoString += f[i]
                
        elif infoType in ['M_float', 'P_float', 'C_float']:
            acceptedChar = [str(i) for i in range(10)] + ['-']
            string = '_' + infoType[0]
            iStart = re.search(string, f).end()
            i = iStart
            infoString = '' + f[i]
            while f[i+1] in acceptedChar and i < len(f)-1:
                i += 1
                infoString += f[i]
            infoString = infoString.replace('-', '.')
            infoString = float(infoString)
                
        elif infoType == 'date':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            infoString = date
        
        elif infoType == 'manipID':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            manip = 'M' + findInfosInFileName(f, 'M')
            infoString = date + '_' + manip
            
        elif infoType == 'cellName':
            infoString = 'M' + findInfosInFileName(f, 'M') + \
                         '_' + 'P' + findInfosInFileName(f, 'P') + \
                         '_' + 'C' + findInfosInFileName(f, 'C')
            
        elif infoType == 'cellID':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            infoString = date + '_' + 'M' + findInfosInFileName(f, 'M') + \
                                '_' + 'P' + findInfosInFileName(f, 'P') + \
                                '_' + 'C' + findInfosInFileName(f, 'C')
                                
        elif infoType == 'substrate':
            try:
                pos = re.search(r"disc[\d]*um", f)
                infoString = f[pos.start():pos.end()]
            except:
                infoString = ''
                
        elif infoType == 'ordinalValue':
            M, P, C = findInfosInFileName(f, 'M'), findInfosInFileName(f, 'P'), findInfosInFileName(f, 'C')
            L = [M, P, C]
            for i in range(len(L)):
                s = L[i]
                if '-' in s:
                    s = s.replace('-', '.')
                    L[i] = s
            [M, P, C] = L
            ordVal = int(float(M)*1e9 + float(P)*1e6 + float(C)*1e3)
            infoString = str(ordVal)
            
    except:
        pass
                             
    return(infoString)

def isFileOfInterest(f, manips, wells, cells, mode = 'soft', suffix = ''):
    """
    Determine if a file f correspond to the given criteria.
    More precisely, return a boolean saying if the manip, well and cell number are in the given range.
    f is a file name. Each of the fields 'manips', 'wells', 'cells' can be either a number, a list of numbers, or 'all'.
    Example : if f = '21-01-18_M2_P1_C8.tif'
    * manips = 'all', wells = 'all', cells = 'all' -> the function return True.
    * manips = 1, wells = 'all', cells = 'all' -> the function return False.
    * manips = [1, 2], wells = 'all', cells = 'all' -> the function return True.
    * manips = [1, 2], wells = 2, cells = 'all' -> the function return False.
    * manips = [1, 2], wells = 1, cells = [5, 6, 7, 8] -> the function return True.
    Example2 : if f = '21-01-18_M2_P1_C8-1.tif'
    * manips = [1, 2], wells = 1, cells = [5, 6, 7, 8] -> the function return False.
    * manips = [1, 2], wells = 1, cells = [5, 6, 7, '8-1'] -> the function return True.
    Note : if manips = 'all', the code will consider that wells = 'all', cells = 'all'.
           if wells = 'all', the code will consider that cells = 'all'.
           This means you can add filters only in this order : manips > wells > cells.
    """
    test = False
    testM, testP, testC = False, False, False
    testSuffix = False
    
    listManips, listWells, listCells = toListOfStrings(manips), toListOfStrings(wells), toListOfStrings(cells)
    
    try:
        fM = (findInfosInFileName(f, 'M'))
        fP = (findInfosInFileName(f, 'P'))
        fC = (findInfosInFileName(f, 'C'))
    except:
        return(False)
    
    if mode == 'soft':
        L = [fM, fP, fC]
        for i in range(3):
            x = L[i]
            try:
                L[i] = x.split('-')[0]
            except:
                pass
        fM, fP, fC = L
            
    elif mode == 'strict':
        pass
    else:
        pass
    
    if (manips == 'all') or (fM in listManips):
        testM = True
    if (wells == 'all') or (fP in listWells):
        testP = True
    if (cells == 'all') or (fC in listCells):
        testC = True
    if (suffix == '') or f.endswith(suffix):
        testSuffix = True
        
    test = (testM and testP and testC and testSuffix)
    return(test)


def simplifyCellId(f):
    res = f
    templateStr = r'M[0-9]_P[0-9]_C[0-9\-]+'
    s = re.search(templateStr, f)
    if s:
        iStart, iStop = s.span()
        foundStr = f[iStart:iStop]
        newStr = foundStr.split('-')[0]
        res = res[:iStart] + newStr + res[iStop:]
    return(res)


def getDictAggMean(df):
    dictAggMean = {}
    for c in df.columns:
        # print(c)
        S = df[c].dropna()
        lenNotNan = S.size
        if lenNotNan == 0:
            dictAggMean[c] = 'first'
        else:
            S.infer_objects()
            if S.dtype == bool:
                dictAggMean[c] = np.nanmin
            else:
                # print(c)
                # print(pd.Series.all(S.apply(lambda x : isinstance(x, numbers.Number))))
                if pd.Series.all(S.apply(lambda x : isinstance(x, numbers.Number))):
                    dictAggMean[c] = np.nanmean
                else:
                    dictAggMean[c] = 'first'
                    
    if 'compNum' in dictAggMean.keys():
        dictAggMean['compNum'] = np.nanmax
                    
    return(dictAggMean)


def flattenPandasIndex(pandasIndex):
    """
    Flatten a multi-leveled pandas index.

    Parameters
    ----------
    pandasIndex : pandas MultiIndex
        Example: MultiIndex([('course', ''),('A', 'count'),('A', 'sum'),( 'coeff', 'sum')])

    Returns
    -------
    new_pandasIndex : list that can be reasigned as a flatten index.
        In the former example: new_pandasIndex = ['course', 'A_count', 'A_sum', 'coeff_sum']
        
    Example
    -------
    >>> data_agg.columns
    >>> Out: MultiIndex([('course',      ''),
    >>>                 (     'A', 'count'),
    >>>                 (     'A',   'sum'),
    >>>                 ( 'coeff',   'sum')],
    >>>                )
    >>> data_agg.columns = flattenPandasIndex(data_agg.columns)
    >>> data_agg.columns
    >>> Out: Index(['course', 'A_count', 'A_sum', 'coeff_sum'], dtype='object')

    """
    new_pandasIndex = []
    for idx in pandasIndex:
        new_idx = '_'.join(idx)
        while new_idx[-1] == '_':
            new_idx = new_idx[:-1]
        new_pandasIndex.append(new_idx)
    return(new_pandasIndex)


def importTrackMateTracks(filepath):
    """
    Parse a TrackMate XML file and return list of tracks.
    Each track: numpy array [t, x, y].
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    tracks = []
    for particle in root.findall('particle'):
        L = []
        for detection in particle.iter("detection"):
            # print(detection)
            # ID = int(spot.attrib["ID"])
            t = float(detection.attrib["t"])
            x = float(detection.attrib["x"])
            y = float(detection.attrib["y"])
            L.append([t, x, y])
        tracks.append(np.array(L))
    return(tracks)


def trackSelection(tracks, mode="longest"):
    """
    Select one track.
    mode="longest": pick the track with most frames
    """
    if mode == "longest":
        lengths = [len(tr) for tr in tracks]
        idx = int(np.argmax(lengths))
        return(tracks[idx])
    else:
        print('Unspecified track selection')
        # Fallback: first track
        return(tracks[0])


# %%% File manipulation

def copyFile(DirSrc, DirDst, filename):
    """
    Copy the file 'filename' from 'DirSrc' to 'DirDst'
    """
    PathSrc = os.path.join(DirSrc, filename)
    PathDst = os.path.join(DirDst, filename)
    print(PathDst)
    shutil.copyfile(PathSrc, PathDst)
    
def copyFolder(DirSrc, DirDst, folderName):
    """
    Copy the folder 'folderName' from 'DirSrc' to 'DirDst' with all its contents
    """
    PathSrc = os.path.join(DirSrc, folderName)
    PathDst = os.path.join(DirDst, folderName)
    shutil.copytree(PathSrc, PathDst)
    
def copyFilesWithString(DirSrc, DirDst, stringInName):
    """
    Copy all the files from 'DirSrc' which names contains 'stringInName' to 'DirDst'
    """
    SrcFilesList = os.listdir(DirSrc)
    for SrcFile in SrcFilesList:
        if stringInName in SrcFile:
            print(SrcFile)
            copyFile(DirSrc, DirDst, SrcFile)
            
def containsFilesWithExt(Dir, ext):
    answer = False
    FilesList = os.listdir(Dir)
    for File in FilesList:
        if File.endswith(ext):
            answer = True
    return(answer)

def getFileNamesWithExt(Dir, ext):
    FilesList = os.listdir(Dir)
    output = [f for f in FilesList if f.endswith(ext)]
    return(output)

def softMkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

# %%% Tiff files

# fileName = '25-12-18_20x_FastBFGFP_1_MMStack_Default.ome.tif'

#### Made in PMMH

def get_CZT_fromTiff(filePath):
    img = Image.open(filePath)
    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
    # md_str = str(meta_dict['ImageJMetaData'].decode('UTF-8'))
    md_str = str(meta_dict['ImageJMetaData'].decode('UTF-8'))[1:-3:2]
    md_array = np.array(md_str.split('#')).astype('<U100')
    channel_str = r'c:\d+/\d+'
    slice_str = r'z:\d+/\d+'
    time_str = r't:\d+/\d+'
    line0 = md_array[0]
    c_match = re.search(channel_str, line0)
    c_tot = int(line0[c_match.start():c_match.end():].split('/')[-1])
    z_match = re.search(slice_str, line0)
    z_tot = int(line0[z_match.start():z_match.end():].split('/')[-1])
    t_match = re.search(time_str, line0)
    t_tot = int(line0[t_match.start():t_match.end():].split('/')[-1])
    czt_shape = [c_tot, z_tot, t_tot]

    czt_seq = []
    for line in md_array:
        if len(line) > 12:
            c_match = re.search(channel_str, line)
            c_raw = line[c_match.start():c_match.end():]
            c = c_raw.split('/')[0].split(':')[-1]
            z_match = re.search(slice_str, line)
            z_raw = line[z_match.start():z_match.end():]
            z = z_raw.split('/')[0].split(':')[-1]
            t_match = re.search(time_str, line)
            t_raw = line[t_match.start():t_match.end():]
            t = t_raw.split('/')[0].split(':')[-1]
            czt_seq.append([int(c), int(z), int(t)])
            
    return(czt_shape, czt_seq)

def OMEReadField(parsed_str, target_str):
    target_str = r'' + target_str
    res = []
    matches = re.finditer(target_str, parsed_str)
    for m in matches:
        str_num = parsed_str[m.end():m.end()+30]
        m_num = re.search(r'[\d\.]+', str_num)
        val_num = float(str_num[m_num.start():m_num.end()])
        res.append(val_num)
    return(res)


def OMEDataParser_tz(filepath):
    # with open(filepath, 'r') as f:
    #     text = f.read()
    
    ome = from_tiff(filepath)
    text = ome.to_xml()
        
    nC, nT, nZ = OMEReadField(text, ' SizeC=')[0], OMEReadField(text, ' SizeT=')[0], OMEReadField(text, ' SizeZ=')[0]
    nC, nT, nZ = int(nC), int(nT), int(nZ)
    CTZ_tz = np.zeros((nC, nT, nZ, 2))
    
    lines = text.split('\n')
    plane_lines = []
    for line in lines:
        if line.startswith('<Plane'):
            plane_lines.append(line)
    
    for line in plane_lines:
        cIdx = int(OMEReadField(line, r' TheC=')[0])
        tIdx = int(OMEReadField(line, r' TheT=')[0])
        zIdx = int(OMEReadField(line, r' TheZ=')[0])
        
        tVal = OMEReadField(line, r' DeltaT=')[0]
        zVal = OMEReadField(line, r' PositionZ=')[0]
        
        CTZ_tz[cIdx, tIdx, zIdx] = [tVal, zVal]
    
    return(CTZ_tz)

def OMEDataParser_t(filepath):
    # with open(filepath, 'r') as f:
    #     text = f.read()
    
    ome = from_tiff(filepath)
    text = ome.to_xml()
        
    nC, nT, nZ = OMEReadField(text, ' SizeC=')[0], OMEReadField(text, ' SizeT=')[0], OMEReadField(text, ' SizeZ=')[0]
    nC, nT, nZ = int(nC), int(nT), int(nZ)
    CTZ_t = np.zeros((nC, nT, nZ))
    
    lines = text.split('\n')
    plane_lines = []
    for line in lines:
        if line.startswith('<Plane'):
            plane_lines.append(line)
    
    for line in plane_lines:
        cIdx = int(OMEReadField(line, r' TheC=')[0])
        tIdx = int(OMEReadField(line, r' TheT=')[0])
        zIdx = int(OMEReadField(line, r' TheZ=')[0])
        
        tVal = OMEReadField(line, r' DeltaT=')[0]
        
        print(tVal)
        
        CTZ_t[cIdx, tIdx, zIdx] = tVal
    
    return(CTZ_t)

#### WORK IN PROGRESS !

WD_path = 'C:/Users/Joseph/Desktop/WorkingData'    
# WD_path = 'E:/WorkingData'
dirPath = WD_path + '/LeicaData/26-01-14_UVinLiveCells/D1_MyOne_200mM-I2959_20pHPMA/26-01-13_C3_beforeUV_1'
filePath = dirPath + '/26-01-13_C3_beforeUV_1_MMStack_Default.ome.tif'
# dirPath = 'E:/WorkingData/LeicaData/25-12-18_WithJessica/25-12-18_Droplet01_JN-Magnet_MyOne-Gly80/'
# print(extractDT(dirPath))

CTZ_t = OMEDataParser_t(filePath)
# ome = from_tiff(filePath)
# xmlText = ome.to_xml()

#### Made in IJM

def extractDT(dirPath):
    S = '{http://www.openmicroscopy.org/Schemas/OME/2016-06}'
    fileNames = os.listdir(dirPath)
    T = []
    C = []
    
    for fN in fileNames:
        if fN.endswith('.tif'):
            filePath = os.path.join(dirPath, fN)
            ome = from_tiff(filePath)
            xmlText = ome.to_xml()
            root = ET.fromstring(xmlText)
            for image in root.findall(S + 'Image'):
                for plane in image.iter(S + "Plane"):
                    c = float(plane.attrib["TheC"])
                    t = float(plane.attrib["DeltaT"])
                    C.append(c)
                    T.append(t)
        break
    
    C = np.array(C)
    T = np.array(T)
    idx_BF = (C == 0)
    T_BF = T[idx_BF]
    dT = T_BF[1:] - T_BF[:-1]
    mean_dT = np.mean(dT)
    median_dT = np.median(dT)
    std_dT = np.std(dT)
    
    return(mean_dT, median_dT, std_dT)


def getTimeFromTiff(path):
    S = '{http://www.openmicroscopy.org/Schemas/OME/2016-06}'
    T = []
    C = []
    
    if path.endswith('.tif'):
        filePath = path
        ome = from_tiff(filePath)
        xmlText = ome.to_xml()
        root = ET.fromstring(xmlText)
        for image in root.findall(S + 'Image'):
            for plane in image.iter(S + "Plane"):
                c = float(plane.attrib["TheC"])
                t = float(plane.attrib["DeltaT"])
                C.append(c)
                T.append(t)
    
    C = np.array(C)
    T = np.array(T)
    idx_C0 = (C == 0)
    T_C0 = T[idx_C0]
    dT_C0 = T_C0[1:] - T_C0[:-1]
    # mean_dT = np.mean(dT)
    # median_dT = np.median(dT)
    # std_dT = np.std(dT)
    
    return(T_C0, dT_C0)


# dirPath = 'E:/WorkingData/LeicaData/25-12-18_WithJessica/25-12-18_Capi01_JN-Magnet_MyOne-Gly80/25-12-18_20x_FastBFGFP_1'
# dirPath = 'E:/WorkingData/LeicaData/25-12-18_WithJessica/25-12-18_Droplet01_JN-Magnet_MyOne-Gly80/'
# print(extractDT(dirPath))

# dirPath = 'E:/WorkingData/LeicaData/26-01-07_Calib_MagnetJingAude_20x_MyOneGly75p_Capillary/26-01-07_20x_MyOneGly75p_Capillary_1'
# print(extractDT(dirPath))
        
def checkDT_multiFile(dirPath):        
    list_dir = os.listdir(dirPath)
    D = {'file name':[], 'median dT':[], 'mean dT':[], 'std dT':[]}
    for d in list_dir:
        p = os.path.join(dirPath, d)
        if os.path.isdir(p):
            lf = os.listdir(p)
            for f in lf:
                if f.endswith('.tif'):
                    pf = os.path.join(p, f)
                    T_C0, dT_C0 = getTimeFromTiff(pf)
                    D['file name'].append(f)
                    D['median dT'].append(np.median(dT_C0))
                    D['mean dT'].append(np.mean(dT_C0))
                    D['std dT'].append(np.std(dT_C0))
    for k in D.keys():
        D[k] = np.array(D[k])
    df = pd.DataFrame(D)
    return(df)

# dirPath = 'E:/WorkingData/LeicaData/26-01-14_UVinLiveCells/D1_MyOne_200mM-I2959_20pHPMA'
# df = checkDT_multiFile(dirPath)
            

# %%% Stats

def get_R2(Ymeas, Ymodel):
    meanY = np.mean(Ymeas)
    meanYarray = meanY*np.ones(len(Ymeas))
    SST = np.sum((Ymeas-meanYarray)**2)
    SSR = np.sum((Ymodel-Ymeas)**2)
    if pd.isnull(SST) or pd.isnull(SSR) or SST == 0:
        R2 = np.nan
    else:
        R2 = 1 - (SSR/SST)
    return(R2)

def get_Chi2(Ymeas, Ymodel, dof, err):
    if dof <= 0:
        Chi2_dof = np.nan
    else:
        residuals = Ymeas-Ymodel
        Chi2 = np.sum((residuals/err)**2)
        Chi2_dof = Chi2/dof
    return(Chi2_dof)

# %%% Image processing


def max_entropy(data):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    M. Emre Celebi
    06.15.2007
    Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
    2016-04-28: Adapted for Python 2.7 by Robert Metchev from Java source of MaxEntropy() in the Autothresholder plugin
    http://rsb.info.nih.gov/ij/plugins/download/AutoThresholder.java
    :param data: Sequence representing the histogram of the image
    :return threshold: Resulting maximum entropy threshold
    """

    # calculate CDF (cumulative density function)
    cdf = data.astype(np.float).cumsum()

    # find histogram's nonzero area
    valid_idx = np.nonzero(data)[0]
    first_bin = valid_idx[0]
    last_bin = valid_idx[-1]

    # initialize search for maximum
    max_ent, threshold = 0, 0

    for it in range(first_bin, last_bin + 1):
        # Background (dark)
        hist_range = data[:it + 1]
        hist_range = hist_range[hist_range != 0] / cdf[it]  # normalize within selected range & remove all 0 elements
        tot_ent = -np.sum(hist_range * np.log(hist_range))  # background entropy

        # Foreground/Object (bright)
        hist_range = data[it + 1:]
        # normalize within selected range & remove all 0 elements
        hist_range = hist_range[hist_range != 0] / (cdf[last_bin] - cdf[it])
        tot_ent -= np.sum(hist_range * np.log(hist_range))  # accumulate object entropy

        # find max
        if tot_ent > max_ent:
            max_ent, threshold = tot_ent, it

    return(threshold)

def max_entropy_threshold(I):
    """
    Function based on the previous one that directly takes an image for argument.
    """
    H, bins = exposure.histogram(I, nbins=256, source_range='image', normalize=False)
    T = max_entropy(H)
    return(T)


def resize_2Dinterp(I, new_nx=None, new_ny=None, fx=None, fy=None):
    
    nX, nY = I.shape[1], I.shape[0]
    X, Y = np.arange(0, nX, 1), np.arange(0, nY, 1)
    try:
        newX, newY = np.arange(0, nX, nX/new_nx), np.arange(0, nY, nY/new_ny)
    except:
        newX, newY = np.arange(0, nX, 1/fx), np.arange(0, nY, 1/fy)
        
    # print(X.shape, Y.shape, newX.shape, newY.shape, I.shape)
    # fd = interpolate.interp2d(XX, ZZ, deptho, kind='cubic')
    # depthoHD = fd(XX, ZZ_HD)
    
    fd = interpolate.RectBivariateSpline(Y, X, I)
    newYY, newXX = np.meshgrid(newY, newX, indexing='ij')
    new_I = fd(newYY, newXX, grid=False)
    return(new_I)


def fitCircle(contour, loss = 'huber'):
    """
    Find the best fitting circle to a an array of points in 2D.
    The contour doesn't have to be the whole circle, it can be simply an arc.

    Parameters
    ----------
    contour : Array-like
        Shape (N, 2). Format RC (Row-Column), which means YX.
        contour = [[Y1, X1], [Y2, X2], [Y3, X3], ...] = [[R1, C1], [R2, C2], [R3, C3], ...]
    loss : string, optional
        Type of loss function applied by least_squares. The default is 'huber', for a robust fit. For a normal least square fit, use 'linear'.
        See documentation on https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    Returns
    -------
    center : tuple (Y, X), center of the circle
    R : float, radius of the circle

    """
    # Contour = [[Y, X], [Y, X], [Y, X], ...] 
    x, y = contour[:,1], contour[:,0]
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return(((x-xc)**2 + (y-yc)**2)**0.5)


    def f_2(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return(Ri - np.mean(Ri))

    center_estimate = x_m, y_m
    result = least_squares(f_2, center_estimate, loss=loss) # Functions from the scipy.optimize library
    center = result.x
    R = np.mean(calc_R(*center))
    
    return(center, R)

# %%% Physics



def computeMag_M270(B, k_batch = 1):
    M = k_batch * 0.74257*1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
    return(M)

def computeMag_M450(B, k_batch = 1):
    M = k_batch * 1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
    return(M)

def computeForce_M450(B, D, d):
    M = computeMag_M450(B)
    R = D/2
    V = (4*np.pi/3)*(R**3)
    m = M*V
    dist = D + d
    F = (3e5 * 2 * m**2) / (dist**4)
    # plt.plot(B, M)
    return(F)

def plotForce(d = 200e-9):
    fig, axes = plt.subplots(1, 2, figsize = (17/gs.cm_in, 6/gs.cm_in)) 
    ax = axes[0]
    B = np.linspace(1, 1000, 1000)
    D = 4500e-9
    F = computeForce_M450(B, D, d)
    ax.plot(B, F)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('B (mT)')
    ax.set_ylabel('F (pN)')
    
    ax = axes[1]
    B = np.linspace(0, 100, 101)
    D = 4500e-9
    F = computeForce_M450(B, D, d)
    ax.plot(B, F)
    ax.set_xlabel('B (mT)')
    ax.set_ylabel('F (pN)')
    
    fig.suptitle('F = f(B) for beads with R={:.1f}µm and d={:.0f}nm'.format(D*1e6, d*1e9))
    plt.tight_layout()
    plt.show()
    
    #### Save
    # figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"
    # figSubDir = 'Mat&Meth'
    # name = 'Force_vs_MagField'
    # ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
    #                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    
def plotMandForce(d = 0):
    D = 4500e-9
    gs.set_manuscript_options_jv()
    
    fig, axes = plt.subplots(2, 2, figsize = (12.5/gs.cm_in, 8.5/gs.cm_in),
                             sharex = 'col', sharey = 'row') 
    
    #### 1.
    B = np.linspace(0, 100, 1000)
    M = computeMag_M450(B, k_batch = 1)
    F = computeForce_M450(B, D, d)
    
    ax = axes[0,0]
    ax.plot(B, M/1e3, c='indigo')
    # ax.set_xlabel('B (mT)')
    ax.set_ylabel('M (kA/m)')
    ax.grid(axis='both')
    ax.set_ylim([-2,32])
    
    ax = axes[1,0]
    # ax.plot(B, F/1e3, c='darkred')
    ax.plot(B, F, c='darkred')
    ax.set_xlabel('B (mT)')
    ax.set_ylabel('F (nN)')
    ax.grid(axis='both')
    ax.set_ylim([-0.2,3.2])
    
    #### 1.
    B = np.linspace(0, 500, 1000)
    M = computeMag_M450(B, k_batch = 1)
    F = computeForce_M450(B, D, d)
    
    ax = axes[0,1]
    ax.plot(B, M/1e3, c='indigo')
    # ax.set_xlabel('B (mT)')
    # ax.set_ylabel('M (pN)')
    ax.grid(axis='both')    
    
    ax = axes[1,1]
    # ax.plot(B, F/1e3, c='darkred')
    ax.plot(B, F, c='darkred')
    ax.set_xlabel('B (mT)')
    # ax.set_ylabel('F (pN)')
    ax.grid(axis='both')
    
    # fig.suptitle('F = f(B) for beads with R={:.1f}µm and d={:.0f}nm'.format(D*1e6, d*1e9))
    plt.tight_layout()
    plt.show()
    
    #### Save
    # figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"
    # figSubDir = 'Mat&Meth'
    # name = 'Force_vs_MagField'
    # archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
    #                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# plotMandForce(d = 0)

def plotForce_Insert(d = 0):
    D = 4500e-9
    gs.set_manuscript_options_jv()
    
    fig, axes = plt.subplots(1, 1, figsize = (3*1.5/gs.cm_in, 2*1.5/gs.cm_in),
                             sharex = 'col', sharey = 'row') 
    
    #### 1.
    B = np.linspace(0, 100, 1000)
    M = computeMag_M450(B, k_batch = 1)
    F = computeForce_M450(B, D, d)
    
    # ax = axes[0,0]
    # ax.plot(B, M/1e3, c='indigo')
    # # ax.set_xlabel('B (mT)')
    # ax.set_ylabel('M (kA/m)')
    # ax.grid(axis='both')
    # ax.set_ylim([-2,32])
    
    # ax = axes[1,0]
    # ax.plot(B, F/1e3, c='darkred')
    # ax.set_xlabel('B (mT)')
    # ax.set_ylabel('F (nN)')
    # ax.grid(axis='both')
    # ax.set_ylim([-0.2,3.2])
    
    #### 1.
    B = np.linspace(0, 10, 500)
    M = computeMag_M450(B, k_batch = 1)
    F = computeForce_M450(B, D, d)
    
    # ax = axes[0,1]
    # ax.plot(B, M/1e3, c='indigo')
    # # ax.set_xlabel('B (mT)')
    # # ax.set_ylabel('M (pN)')
    # ax.grid(axis='both')    
    
    ax = axes#[1,1]
    ax.plot(B, F/1e3, c='darkred')
    # ax.set_xlabel('B (mT)')
    # ax.set_ylabel('F (pN)')
    ax.grid(axis='both')
    ax.tick_params(axis=u'both', which=u'both',length=0, labelsize=8)
    
    # fig.suptitle('F = f(B) for beads with R={:.1f}µm and d={:.0f}nm'.format(D*1e6, d*1e9))
    plt.tight_layout()
    plt.show()
    
    #### Save
    figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"
    figSubDir = 'Mat&Meth'
    name = 'Force_vs_MagField_insert'
    plt.show()
    # archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
    #                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# plotForce_Insert(d = 0)

 
                    
# %%% Very general functions

def findFirst(x, A):
    """
    Find first occurence of x in array A, in a VERY FAST way.
    If you like weird one liners, you will like this function.
    """
    idx = (A==x).view(bool).argmax()
    return(idx)

def findLast(x, A):
    """
    Find last occurence of x in array A, in a VERY FAST way.
    Adapted from findFirst just above.
    """
    idx = (A[::-1]==x).view(bool).argmax()
    return(len(A)-idx-1)


def findFirst_V2(v, arr):
    """
    https://stackoverflow.com/questions/432112/is-there-a-numpy-function-to-return-the-first-index-of-something-in-an-array
    """
    for idx, val in np.ndenumerate(arr):
        if val == v:
            return(idx[0])
    return(-1)


def argmedian(x):
    """
    Find the argument of the median value in array x.
    """
    if len(x)%2 == 0:
        x = x[:-1]
    return(np.argpartition(x, len(x) // 2)[len(x) // 2])


def sortMatrixByCol(A, col=0, direction = 1):
    """
    Return the matrix A where all columns where sorted by the column i
    Increasing order if direction = 1
    Decreasing order if direction = -1
    """
    return(A[A[:, col].argsort()[::direction]])



def fitLine(X, Y):
    """
    returns: results.params, results \n
    Y=a*X+b ; params[0] = b,  params[1] = a
    
    NB:
        R2 = results.rsquared \n
        ci = results.conf_int(alpha=0.05) \n
        CovM = results.cov_params() \n
        p = results.pvalues \n
    
    This is how one should compute conf_int:
        bse = results.bse \n
        dist = stats.t \n
        alpha = 0.05 \n
        q = dist.ppf(1 - alpha / 2, results.df_resid) \n
        params = results.params \n
        lower = params - q * bse \n
        upper = params + q * bse \n
    """
    
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    params = results.params 
#     print(dir(results))
    return(results.params, results)


def fitLineHuber(X, Y, with_wlm_results = False):
    """
    returns: results.params, results \n
    Y=a*X+b ; params[0] = b,  params[1] = a
    
    NB:
        R2 = results.rsquared \n
        ci = results.conf_int(alpha=0.05) \n
        CovM = results.cov_params() \n
        p = results.pvalues \n
    
    This is how one should compute conf_int:
        bse = results.bse \n
        dist = stats.t \n
        alpha = 0.05 \n
        q = dist.ppf(1 - alpha / 2, results.df_resid) \n
        params = results.params \n
        lower = params - q * bse \n
        upper = params + q * bse \n
    """
    
    X = sm.add_constant(X)
    model = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
    results = model.fit()
    params = results.params
    
    if not with_wlm_results:
        out = (results.params, results)
    else:
        weights = results.weights
        w_model = sm.WLS(Y, X, weights)
        w_results = w_model.fit()
        out = (results.params, results, w_results)
    return(out)


def fitLineTLS(X, Y):
    """

    """
    def linearFun(B, X):
        return(B[0]*X + B[1])
    linear = odr.Model(linearFun)
    data = odr.Data(X, Y, wd=1, we=1)
    fit = odr.ODR(data, linear, beta0=[0, 0])
    output = fit.run()
    a, b = output.beta

    out = ((a, b), output)
    
    return(out)


def fitLineWeighted(X, Y, weights):
    """
    returns: results.params, results \n
    Y=a*X+b ; params[0] = b,  params[1] = a
    
    NB:
        R2 = results.rsquared \n
        ci = results.conf_int(alpha=0.05) \n
        CovM = results.cov_params() \n
        p = results.pvalues \n
    
    This is how one should compute conf_int:
        bse = results.bse \n
        dist = stats.t \n
        alpha = 0.05 \n
        q = dist.ppf(1 - alpha / 2, results.df_resid) \n
        params = results.params \n
        lower = params - q * bse \n
        upper = params + q * bse \n
    """
    
    X = sm.add_constant(X)
    model = sm.WLS(Y, X, weights)
    results = model.fit()
    params = results.params 

    return(results.params, results)


def toList(x):
    """
    if x is a list, return x
    if x is not a list, return [x]
    
    Reference
    ---------
    https://docs.python.org/3/library/collections.abc.html
    """
    t1 = isinstance(x, str) # Test if x is a string
    if t1: # x = 'my_string'
        return([x]) # return : ['my_string']
    else:
        t2 = isinstance(x, Collection) # Test if x is a Collection
        if t2: # x = [1,2,3] or x = array([1, 2, 3]) or x = {'k1' : v1}
            return(x) # return : x itself
        else: # x is not a Collection : probably a number or a boolean
            return([x]) # return : [x]
        
        
def toListOfStrings(x):
    """
    if x is a list, return x with all elements converted to string
    if x is not a list, return ['x']
    
    Reference
    ---------
    https://docs.python.org/3/library/collections.abc.html
    """
    t1 = isinstance(x, str) # Test if x is a string
    if t1: # x = 'my_string'
        return([x]) # return : ['my_string']
    else:
        t2 = isinstance(x, Collection) # Test if x is a Collection
        if t2: # x = [1,2,3] or x = array([1, 2, 3]) or x = {'k1' : v1}
            xx = [str(xi) for xi in x]
            return(xx) # return : x itself
        else: # x is not a Collection : probably a number or a boolean
            return([str(x)]) # return : [x]
        

    
def drop_duplicates_in_array(A):
    val, idx = np.unique(A, return_index = True)
    idx.sort()
    A_filtered = np.array([A[idx[j]] for j in range(len(idx))])
    return(A_filtered)



def interDeciles(A):
    p10 = np.percentile(A, 10)
    p90 = np.percentile(A, 90)
    return(p90-p10)


def strToMask(A, S):
    LS = S.split('_')
    kind = LS[1]
    if kind == '<':
        upperBound = float(LS[2])
        mask = (A < upperBound)
    elif kind == '>':
        lowerBound = float(LS[2])
        mask = (A > lowerBound)
    elif kind == 'in':
        lowerBound = float(LS[2])
        upperBound = float(LS[3])
        mask = (A < upperBound) & (A > lowerBound)
    return(mask)



# %%% Figure & graphic operations

def simpleSaveFig(fig, name, savePath, ext, dpi):
    figPath = os.path.join(savePath, name + ext)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    fig.savefig(figPath, dpi=dpi)
    

#### def archiveFig(fig, name = '', ext = '.pdf', dpi = 150,
    #            figDir = '', figSubDir = '', cloudSave = 'flexible'):
    # """
    # This is supposed to be a "smart" figure saver.
    
    # 1. Main save
    #     - It saves the fig with resolution 'dpi' and extension 'ext' (default ext = '.png' and dpi = 100).
    #     - If you give a name, it will be used to save your file; if not, a name will be generated based on the date.
    #     - If you give a value for figDir, your file will be saved in cp.DirDataFig//figDir. Else, it will be in cp.DirDataFigToday.
    #     - You can also give a value for figSubDir to save your fig in a subfolder of the chosen figDir.
    
    # 2. Backup save (optional). cloudSave can have 3 values : 'strict', 'flexible', or 'none'.
    #     - If 'strict', this function will attempt to do a cloud save not matter what.
    #     - If 'check', this function will check that you enable properly the cloud save in CortexPath before attempting to do a cloud save.
    #     - If 'none', this function will not do a cloud save.
    # """
    # # Generate unique name if needed
    # if name == '':
    #     dt = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    #     name = 'fig_' + dt
    # # Generate default save path if needed
    # if figDir == '':
    #     figDir = cp.DirDataFigToday
    #     figCloudDir = cp.DirCloudFigToday
    # else:
    #     figDir = os.path.join(cp.DirDataFig, figDir)
    #     figCloudDir = os.path.join(cp.DirCloudFig, figDir)
    # # Normal save
    # savePath = os.path.join(figDir, figSubDir)
    # simpleSaveFig(fig, name, savePath, ext, dpi)
    # # Cloud save if specified
    # doCloudSave = ((cloudSave == 'strict') or (cloudSave == 'flexible' and cp.CloudSaving != ''))
    # if doCloudSave:
    #     cloudSavePath = os.path.join(figCloudDir, figSubDir)
    #     simpleSaveFig(fig, name, cloudSavePath, ext, dpi)
        
   
def setCommonBounds(axes, xb = [0, 'auto'], yb = [0, 'auto'],
                    largestX = [-np.inf, np.inf], largestY = [-np.inf, np.inf]):
    
    
    axes_f = axes.flatten()
        
    if xb[0] == 'auto':
        x_min =  np.min([ax.get_xlim()[0] for ax in axes_f])
    else:
        x_min = xb[0]
        
    if xb[1] == 'auto':
        x_max =  np.max([ax.get_xlim()[1] for ax in axes_f])
    else:
        x_max = xb[1]
        
    if yb[0] == 'auto':
        y_min =  np.min([ax.get_ylim()[0] for ax in axes_f])
    else:
        y_min = yb[0]
        
    if yb[1] == 'auto':
        y_max =  np.max([ax.get_ylim()[1] for ax in axes_f])
    else:
        y_max = yb[1]
        
    for ax in axes_f:
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
    
    return(axes)


def setCommonBounds_V2(axes, mode = 'firstLine', xb = [0, 'auto'], yb = [0, 'auto'],
                    Xspace = [-np.inf, np.inf], Yspace = [-np.inf, np.inf]):

    axes_f = axes.flatten()
    if len(axes_f) > 1:
        x_bounds = []
        y_bounds = []
        
        for ax in axes_f:
            lines = ax.get_lines()
            if len(lines) == 0:
                continue
            else:
                if mode == 'firstLine':
                    L = lines[0]
                    data = L.get_data()
                elif mode == 'allLines':
                    data = (np.concatenate([L.get_data()[0] for L in lines]),
                            np.concatenate([L.get_data()[1] for L in lines]))
            
            x_bounds.append([0.85*np.percentile(data[0], 5), 
                             1.15*np.percentile(data[0], 95)])
            y_bounds.append([0.85*np.percentile(data[1], 5), 
                             1.15*np.percentile(data[1], 95)])
        
        x_bounds = np.array(x_bounds)
        y_bounds = np.array(y_bounds)
        
        if len(x_bounds) > 1 and len(y_bounds) > 1:
            X_BOUND = [np.percentile(x_bounds[:,0], 5), np.percentile(x_bounds[:,1], 95)]
            Y_BOUND = [np.percentile(y_bounds[:,0], 5), np.percentile(y_bounds[:,1], 95)]
        
            if xb[0] == 'auto':
                X_BOUND[0] = max(X_BOUND[0], Xspace[0])
            else:
                X_BOUND[0] = xb[0]
                
            if xb[1] == 'auto':
                X_BOUND[1] = min(X_BOUND[1], Xspace[1])
            else:
                X_BOUND[1] = xb[1]
        
        
            if yb[0] == 'auto':
                Y_BOUND[0] =  max(Y_BOUND[0], Yspace[0])
            else:
                Y_BOUND[0] = yb[0]
                
            if yb[1] == 'auto':
                Y_BOUND[1] =  min(Y_BOUND[1], Yspace[1])
            else:
                Y_BOUND[1] = yb[1]
                    
            for ax in axes_f:
                ax.set_xlim(X_BOUND)
                ax.set_ylim(Y_BOUND)
        else:
            pass
    
    # except:
    #      pass 
    
    return(axes)



def setAllTextFontSize(ax, size = 9):
    """
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    size : TYPE, optional
        DESCRIPTION. The default is 9.

    Returns
    -------
    None.

    """
    for item in ([ax.title, ax.xaxis.label, \
                  ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)
    return(ax)
    

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return(colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2]))

# %%% Geometry / design

# %%%% Rheology

def volumeRheoPlanPlan(D, h=1e-3):
    R = D/2
    V = np.pi * R**2 * h
    return(V)

def volumeRheoConePlan(D, alpha, a=0):
    R = D/2
    V1 = (2/3) * np.pi * R**3 * np.tan(alpha*np.pi/180)
    V2 = np.pi * R**2 * a
    V = V1 + V2
    return(V)

#### PP40
# D = 40e-3 # m
# V = volumeRheoPlanPlan(D)
# V_µL = V*1e9 # m3 to µL
# V_µL = 1256.6 # µL

#### CP50-1, a = 0µm
# D = 50e-3 # m
# alpha = 1 # °
# V = volumeRheoConePlan(D, alpha)
# V_µL = V*1e9 # m3 to µL
# V_µL = 571.2 # µL
# print(f'CP{D*1e3:.0f}-{alpha:.0f}, a = {a*1e6:.0f}µm, V = {V_µL:.1f}')

#### CP35-1, a = 30µm
# D = 35e-3 # m
# alpha = 1 # °
# a = 30 * 1e-6
# V = volumeRheoConePlan(D, alpha, a)
# V_µL = V*1e9 # m3 to µL
# # V_µL = 95.9 # µL
# print(f'CP{D*1e3:.0f}-{alpha:.0f}, V = {V_µL:.1f} µL')

#### CP25-1, a = 0µm
# D = 25e-3 # m
# alpha = 1 # °
# V = volumeRheoConePlan(D, alpha)
# V_µL = V*1e9 # m3 to µL
# V_µL = 71.4 # µL
# print(f'CP{D*1e3:.0f}-{alpha:.0f}, a = {a*1e6:.0f}µm, V = {V_µL:.1f}')

#### CP25-1, a = 30µm
# D = 25e-3 # m
# alpha = 1 # °
# a = 30 * 1e-6
# V = volumeRheoConePlan(D, alpha, a)
# V_µL = V*1e9 # m3 to µL
# # V_µL = 95.9 # µL
# print(f'CP{D*1e3:.0f}-{alpha:.0f}, V = {V_µL:.1f} µL')

#### CP25-1, a = 50µm
# D = 25e-3 # m
# alpha = 1 # °
# a = 50 * 1e-6
# V = volumeRheoConePlan(D, alpha, a)
# V_µL = V*1e9 # m3 to µL
# V_µL = 95.9 # µL
# print(f'CP{D*1e3:.0f}-{alpha:.0f}, a = {a*1e6:.0f}µm, V = {V_µL:.1f}')

#### CP25-1, a = 100µm
# D = 25e-3 # m
# alpha = 1 # °
# a = 100 * 1e-6
# V = volumeRheoConePlan(D, alpha, a)
# V_µL = V*1e9 # m3 to µL
# V_µL = 120.5 # µL
# print(f'CP{D*1e3:.0f}-{alpha:.0f}, a = {a*1e6:.0f}µm, V = {V_µL:.1f}')

#### CP25-0.5 , a = 0µm
# D = 25e-3 # m
# alpha = 0.5 # °
# V = volumeRheoConePlan(D, alpha)
# V_µL = V*1e9 # m3 to µL
# V_µL = 35.7 # µL
# print(f'CP{D*1e3:.0f}-{alpha:.0f}, a = {a*1e6:.0f}µm, V = {V_µL:.1f}')

#### CP25-0.5 , a = 15µm
# D = 25e-3 # m
# alpha = 0.5 # °
# a = 15 * 1e-6
# V = volumeRheoConePlan(D, alpha, a)
# V_µL = V*1e9 # m3 to µL
# # V_µL = 60.2 # µL
# print(f'CP{D*1e3:.0f}-{alpha:.1f}, V = {V_µL:.1f} µL')

#### CP25-0.5 , a = 100µm
# D = 25e-3 # m
# alpha = 0.5 # °
# a = 100 * 1e-6
# V = volumeRheoConePlan(D, alpha, a)
# V_µL = V*1e9 # m3 to µL
# V_µL = 84.8 # µL
# print(f'CP{D*1e3:.0f}-{alpha:.0f}, a = {a*1e6:.0f}µm, V = {V_µL:.1f}')

# %%%% Optics

# D1 = 37 * 1e-3 # m
# R1 = D1/2
# S1 = np.pi*R1**2 # m2
# P = 435 * 1e-3 # W

# I1 = P/S1
# I1_mWcm2 = I1 * 1e3 * 1e-4
# print(f'Irradience In: {I1_mWcm2:.2f} mW/cm²')

# Loss_factor = 0.01
# D2 = 460 * 1e-6 # m
# R2 = D2/2
# S2 = np.pi*R2**2 # m2
# I2 = Loss_factor * P/S2
# I2_mWcm2 = I2 * 1e3 * 1e-4
# print(f'Irradience In: {I2_mWcm2:.2f} mW/cm²')

# %% Test

# # %%% Dataset

# Atrue = +1
# Btrue = 0
# XVarTrue = 1.0
# YVarTrue = 5.0

# # Adimension by data spanning? Adimension the variance

# Xtrue = np.arange(start = -10, stop = 11, step = 0.2)
# Ytrue = Atrue*Xtrue + Btrue

# Xr = Xtrue + np.random.normal(loc=0.0, scale=XVarTrue**0.5, size=len(Xtrue))
# Yr = Ytrue + np.random.normal(loc=0.0, scale=YVarTrue**0.5, size=len(Ytrue))

# # %%% Function

# def fitLineTLS(X, Y, wd=1, we=1):
#     """

#     """
#     def linearFun(B, X):
#         return(B[0]*X + B[1])
#     linear = odr.Model(linearFun)
#     data = odr.Data(X, Y, wd=wd, we=we)
#     fit = odr.ODR(data, linear, beta0=[0, 0])
#     output = fit.run()
#     a, b = output.beta
#     out = ((b, a), output)
#     return(out)


# #### Test
# Xplot = np.linspace(-10, 10, num = 100)

# # OLS - X vs Y
# [b_xy, a_xy], results_xy = fitLine(Xr, Yr)
# Yplot_xy = a_xy * Xplot + b_xy

# # OLS - X vs Y
# [B_yx, A_yx], results_yx = fitLine(Yr, Xr)
# a_yx, b_yx = 1/A_yx, -B_yx/A_yx
# Yplot_yx = a_yx * Xplot + b_yx

# # ODR - True variances
# [b_odr1, a_odr1], results_odr1 = fitLineTLS(Xr, Yr, wd=1/XVarTrue, we=1/YVarTrue)
# Yplot_odr1 = a_odr1 * Xplot + b_odr1

# # ODR - Estimated variances
# [b_odr2, a_odr2], results_odr2 = fitLineTLS(Xr, Yr, wd=1, we=1)
# Yplot_odr2 = a_odr2 * Xplot + b_odr2

# # ODR - Force to XvY equivalent ----> Works as intended !
# [b_odr3, a_odr3], results_odr3 = fitLineTLS(Xr, Yr, wd=1, we=0.000001)
# Yplot_odr3 = a_odr3 * Xplot + b_odr3

# fig, ax = plt.subplots(1, 1)
# ax.plot(Xr, Yr, 'ko')
# ax.plot(Xplot, Yplot_xy,   'b--', label=f'XvY | a={a_xy:.2f}')
# ax.plot(Xplot, Yplot_yx,   'r--', label=f'YvX | a={a_yx:.2f}')
# ax.plot(Xplot, Yplot_odr1, 'g-',  label=f'TLS true | a={a_odr1:.2f}')
# ax.plot(Xplot, Yplot_odr2, 'k-',  label=f'TLS est | a={a_odr2:.2f}')
# ax.plot(Xplot, Yplot_odr3, 'c-',  label=f'TLS all y | a={a_odr3:.2f}')
# ax.grid()
# ax.legend(fontsize = 11)


# plt.show()


# %%% User Input

# class MultiChoiceBox(Qtw.QMainWindow):
#     """
#     A class to display a dialog box with multiple choices. Inherited from Qtw.QMainWindow.
#     """
#     def __init__(self, choicesDict, title = 'Multiple choice box'):
#         super().__init__()
        
#         self.choicesDict = choicesDict
#         self.questions = [k for k in choicesDict.keys()]
#         self.nQ = len(self.questions)
        
#         self.answersDict = {}
#         self.list_rbg = [] # rbg = radio button group

#         self.setWindowTitle(title)
        
#         layout = Qtw.QVBoxLayout()  # layout for the central widget
#         main_widget = Qtw.QWidget(self)  # central widget
#         main_widget.setLayout(layout)
        
#         for q in self.questions:
#             choices = self.choicesDict[q]
#             label = Qtw.QLabel(q)
#             layout.addWidget(label)
#             rbg = Qtw.QButtonGroup(main_widget)
#             for c in choices:
#                 rb = Qtw.QRadioButton(c)
#                 rbg.addButton(rb)
#                 layout.addWidget(rb)
#                 if c == choices[0]:
#                     rb.click()
                
#             self.list_rbg.append(rbg)
#             layout.addSpacing(20)
        
#         valid_button = Qtw.QPushButton('OK', main_widget)
#         layout.addWidget(valid_button)
        
#         self.setCentralWidget(main_widget)
        
#         valid_button.clicked.connect(self.validate_button)


#     def validate_button(self):
#         array_err = np.array([rbg.checkedButton() == None for rbg in self.list_rbg])
#         Err = np.any(array_err)
#         if Err:
#             self.error_dialog()
#         else:
#             for i in range(self.nQ):
#                 q = self.questions[i]
#                 rbg = self.list_rbg[i]
#                 self.answersDict[q] = rbg.checkedButton().text()
                
#             self.quit_button()
            
#     def location_on_the_screen(self):
#         ag = Qtw.QDesktopWidget().availableGeometry()
#         # sg = Qtw.QDesktopWidget().screenGeometry()

#         widget = self.geometry()
#         # x = ag.width() - widget.width()
#         # y = 2 * ag.height() - sg.height() - widget.height()
#         x = int(0.15*ag.width())
#         y = int(0.4*ag.height())
#         self.move(x, y)
            
#     def error_dialog(self):
#         dlg = Qtw.QMessageBox(self)
#         dlg.setWindowTitle("Error")
#         dlg.setText("Please make a choice in each category.")
#         dlg.exec()
        
#     def quit_button(self):
#         Qtw.QApplication.quit()
#         self.close()
        
#     def closeEvent(self, event):
#         Qtw.QApplication.quit()


# def makeMultiChoiceBox(choicesDict):
#     """
#     Create and show a dialog box with multiple choices.
    

#     Parameters
#     ----------
#     choicesDict : dict
#         Contains question and possible answers in the form : {Q1 : [A11, A12,...], Q2 : [A21, A22,...], ...},
#         where Qi and Aij are strings.

#     Returns
#     -------
#     answersDict : dict
#         Contains question and chosen answers in the form : {Q1 : A12, Q2 : A25, ...},
#         where Qi and Aij are strings.

#     Example
#     -------
#     >>> choicesDict = {'Is the cell ok?' : ['Yes', 'No'],
#     >>>                'Is the nucleus visible?' : ['Yes', 'No'],}
#     >>> answersDict = makeMultiChoiceBox(choicesDict)
#     >>> print(res)
#     >>> Out: {'Is the cell ok?': 'Yes', 'Is the nucleus visible?': 'No'}

#     """
    
#     app = Qtw.QApplication(sys.argv)
#     MCBox = MultiChoiceBox(choicesDict)
#     MCBox.location_on_the_screen()
#     MCBox.show()
#     app.exec()
    
#     answersDict = MCBox.answersDict
#     return(answersDict)



# choicesDict = {'Is the cell ok?' : ['Yes', 'No', 'peanut butter jelly!'],
#                 'Is the nucleus visible?' : ['Yes', 'No', 'banana!'],}
# answersDict = makeMultiChoiceBox(choicesDict)




