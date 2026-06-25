# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:32:49 2026

@author: Joseph Vermeil

UtilityFunctions.py - contains all kind of small functions used by CortExplore programs, 
to be imported with "import UtilityFunctions as ufun" and call with "ufun.my_function".
Joseph Vermeil, 2026

This program is free software: you can redistribute it and\\or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https:\\\\www.gnu.org\\licenses\\>.
"""


# %% Imports & settings

#### Imports

import os
import re
import cv2
import sys
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

# from trackpy.motion import msd, imsd, emsd
from PIL import Image, ImageDraw
from scipy import signal # stats #, optimize, interpolate, 

import trackpy as tp

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.UtilityFunctions as ufun

#### Settings

SCALE_20X = 0.461
SCALE_40X = 0.229
FPS = 1


# %% Imports 2

import imagej
import scyjava as sj
# import random
sj.config.add_options('-Xmx18g')

os.environ["JAVA_HOME"] = "C:\\Users\\josep\\mambaforge\\envs\\pyimagej-env\\Library\\lib\\jvm"

# initialize ImageJ
# ij = imagej.init()
# ij = imagej.init('sc.fiji:fiji')
ij = imagej.init('C:\\Users\\josep\\Desktop\\Fiji.app\\', add_legacy=True)

print(f"ImageJ version: {ij.getVersion()}")


# %% Helper functions

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



def get_reasonable_inner_cell_contour(img, PLOT = False):
    nT, nY, nX = img.shape
    img_min = np.min(img, axis = 0)
    
    (Yc, Xc), Rc = ufun.find_cell_inner_circle(img_min, binarize = True, 
                                          zero_padding = 10,
                                          PLOT=PLOT)
    Angles = np.linspace(0, 2*np.pi, 360)
    Xcontour = Xc + Rc*np.cos(Angles)
    Ycontour = Yc + Rc*np.sin(Angles)
    contour = np.array([Ycontour, Xcontour]).T
    if PLOT:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img_min, cmap='gray')
        axes[0].plot(contour[:,1], contour[:,0], 'r-')
        mask = ufun.contour_to_mask([nY, nX], contour)
        axes[1].imshow(img[0]*mask, cmap='gray')
        plt.show()
    return(contour)


def get_numbers_following_text(text, target, output = 'integer'):
    if output == 'integer':
        m = re.search(r''+target, text)
        m_num = re.search(r'[\d]+', text[m.end():m.end()+10])
        res = int(text[m.end():m.end()+10][m_num.start():m_num.end()])
    elif output == 'string':
        m = re.search(r''+target, text)
        m_num = re.search(r'[\d-]+', text[m.end():m.end()+10])
        res = str(text[m.end():m.end()+10][m_num.start():m_num.end()])
    return(res)
    


def check_if_file_has_tracks(fileName, srcDir):
    fN_root = fileName.split('.')[0]
    fN_contour = fN_root + '_Tracks.xml'
    has_contours = os.path.isfile(os.path.join(srcDir, fN_contour))
    return(has_contours)



def draw_circles(img, blobs, 
                 fig = None, ax = None):
    # this is the basic function that will be used to draw detected blobs 
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    ax.imshow(img, cmap='gray')
    for blob in blobs:
        y, x, radius = blob
        c = plt.Circle((x, y), radius*np.sqrt(2), color='white', linewidth=2, fill=False)
        ax.add_patch(c)

    # plt.show()  



# %% Main functions


def compute_acor(image, mask, window_length, FPS, 
                 EQUALIZE = True, PLOT = False):
    if EQUALIZE:
        for t in range(image.shape[0]):
            p1, p99 = np.percentile(image[t].flatten()[mask.flatten()], (1, 99))
            image[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
    
    if PLOT:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image[0]*mask, cmap = 'gray')
        axes[1].imshow(image[-1]*mask, cmap = 'gray')
        plt.show()
    
    short_len = window_length
    long_len = image.shape[0] - short_len + 1
    image_acor = np.zeros((long_len, image.shape[1], image.shape[2]))
    
    Zero_std_found = False
    
    image_mean = np.mean(image, axis=0)
    image_std = np.std(image, axis=0)
    non_zero_std = (image_std > 0)
    mask_2 = (mask & non_zero_std)
    image_normalized = (image - image_mean) / (image_std + (1-mask_2))
    
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            if mask_2[i, j]:
                acor = signal.correlate(image_normalized[:,i,j], 
                                        image_normalized[:short_len,i,j], 
                                        mode="valid")
                acor = acor / acor[0]
                image_acor[:, i, j] = acor
                    
    total_acor = np.zeros(long_len)
    lags = np.arange(long_len) * (1/FPS)
    for t in range(len(total_acor)):
        total_acor[t] = np.mean(image_acor[t].flatten()[mask.flatten()])
    
    if PLOT:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(mask, cmap='gray')
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot(lags, total_acor)
        plt.show()
        
    return(total_acor, image_acor)



def analyse_white_blobs_MSD(trackPathList, df_Pa, SCALE, FPS,
                            PLOT = False):
    res_dict = {
                'id':[],
                'pos_id':[],
                'cell_id':[],
                'D':[],
                'k_nl':[],
                'D_nl':[],
                }
    tables_dict = {}
    MSD_dict = {}

    print(pm.BLUE + 'Starting MSD analysis' + pm.NORMAL)    
    
    if PLOT:
        fig, ax = plt.subplots(1, 1)
        Nt = len(trackPathList)
        Nc = len(pm.cL_Set21)
        if Nt <= Nc:
            listColors = pm.cL_Set21[:Nt]
        else:
            listColors = pm.cL_Set21
    
    for k, p in enumerate(trackPathList):
        T0 = time.time()
        
        # Ids
        _, fN = os.path.split(p)
        print(pm.GREEN + f'Analysing {fN}' + pm.NORMAL)
        
        full_id = '_'.join(fN.split('_')[:5])
        manip_id = '_'.join(fN.split('_')[:2])
        pos_id = get_numbers_following_text(fN, '_Pos')
        cell_id = get_numbers_following_text(fN, '_C')
        
        # MSD
        Tracks = importTrackMateTracks(p)
        column_names = ['frame', 'x', 'y', 'particle']
        all_tracks = []
        for i, track in enumerate(Tracks):
            nT = len(track)
            if nT >= 30:
                track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
                track[:,0] = track[:,0].astype(int) + 1
                all_tracks.append(track)
        concat_tracks = np.concat(all_tracks, axis = 0)
        df = pd.DataFrame({column_names[k] : concat_tracks[:,k] for k in range(len(column_names))})
        tables_dict[full_id] = df
        
        #### Run imsd -> Might be useful for SEM computation
        # res_imsd = tp.motion.imsd(df, SCALE, FPS).reset_index()
    
        #### Run msd
        res_emsd = tp.motion.emsd(df, SCALE, FPS, max_lagtime=30).reset_index()
        T, MSD = res_emsd['lagt'], res_emsd['msd']
        MSD_dict[full_id] = np.array([T, MSD]).T
        
        parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
        D = parms.values[0]/4
        
        if PLOT:
            color = listColors[k%Nc]
            dark_color = pm.lighten_color(color, 0.5)
            ax.plot(res_emsd['lagt'], res_emsd['msd'], color=color, marker='.', lw=0.5, 
                    label=full_id)
            ax.axline(xy1=(0,0), slope=D*4, color=dark_color, ls='-', lw=1, 
                      label=f'D = {D:.2e} µm²\\s')
        
        parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
        b, a = parms
        k_nl = a
        D_nl = np.exp(b)/4
        
        res_dict['id'].append(full_id)
        res_dict['pos_id'].append(pos_id)
        res_dict['cell_id'].append(cell_id)
        res_dict['D'].append(D)
        res_dict['k_nl'].append(k_nl)
        res_dict['D_nl'].append(D_nl)
        
        Dt = time.time() - T0
        print(f'Done in Dt = {Dt:.4f}')
        
    if PLOT:
        ax.set_xlabel('Lag times (s)')
        ax.set_ylabel('MSD (µm²)')
        ax.grid()
        ax.legend()
        fig.tight_layout()
        plt.show()
        
    res_df = pd.DataFrame(res_dict)
        
    return(res_df, MSD_dict)



def TrackSpotsInCell(tifPath, dstDir):
    
    SCALE = SCALE_40X
    SIZE_UM = 1.5
    SIZE_PIX = SIZE_UM/SCALE
    print(SIZE_PIX)
    EQUALIZE = True
    N_ERODE = 40
    TOP_HAT = True
    MEDIAN_FILTER = True
    
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Get image and mask
    shape, dtype = ufun.tiff_inspect(tifPath)
    nT = shape[0]
    
    nT_subset = min(100, nT)
    subset_T = np.linspace(0, nT-1, num = nT_subset, dtype=int)
    image_subset = ufun.load_stack_region(tifPath, time_indices=subset_T, 
                                          x_slice=None, y_slice=None)
    image_subset = skm.util.img_as_float32(image_subset)
    
    inner_cell_contour = get_reasonable_inner_cell_contour(image_subset, PLOT = False)
    mask = ufun.contour_to_mask([shape[1], shape[2]], inner_cell_contour)
    mask = ndi.binary_erosion(mask, iterations = N_ERODE)
    
    image = ufun.load_stack_region(tifPath, time_indices=None, 
                                   x_slice=None, y_slice=None)
    image = skm.util.img_as_float32(image)  
    # image = skm.util.img_as_ubyte(image)
    
    # image = image_subset
    # nT = nT_subset
    nT = 1
    
    image_pt = np.zeros_like(image)
    
    #### EQUALIZE
    if EQUALIZE:
        top = time.time()
        for t in range(nT):
            p1, p99 = np.percentile(image[t].flatten()[mask.flatten()], (1, 99))
            image_pt[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
        print(f'Equalize {time.time() - top:.1f} s')    
        
    else:
        for t in range(nT):
            image_pt[t] = image[t, :, :]
    
    #### CROP
    for t in range(nT):
        image_pt[t] = image_pt[t] * mask
            
    ax = axes[0]
    ax.imshow(image_pt[0], cmap='gray')
    
    ax = axes[1]
    ax.imshow(image_pt[0], cmap='gray')
    
    #### FILTER
    if MEDIAN_FILTER:
        top = time.time()
        for t in range(nT):
            k = 3
            image_pt[t] = cv2.medianBlur(image_pt[t], k)
        print(f'Median filter {time.time() - top:.1f} s') 
            
    ax = axes[2]
    ax.imshow(image_pt[0], cmap='gray')
    
    
    if TOP_HAT: # Applying the Top-Hat operation
        top = time.time()
        filterSize = (15, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        for t in range(nT):
            image_pt[t] = cv2.morphologyEx(image_pt[t], cv2.MORPH_TOPHAT, kernel)
            p1, p99 = np.percentile(image_pt[t].flatten()[mask.flatten()], (1, 99))
            image_pt[t] = skm.exposure.rescale_intensity(image_pt[t], in_range=(p1, p99))
        print(f'Top hat {time.time() - top:.1f} s') 
        
    ax = axes[3]
    ax.imshow(image_pt[0], cmap='gray')
    
    
    #### LOCATE
    list_df = []
    
    top = time.time()
    for t in range(nT):
        blobs = skm.feature.blob_dog(image_pt[t], min_sigma=7, max_sigma=7, 
                                     threshold=0.05, overlap=.5, exclude_border=True)
        X = blobs[:, 0]
        Y = blobs[:, 1]
        F = len(X) * [t]
        df = pd.DataFrame({'x':X, 'y':Y, 'frame':F})
        list_df.append(df)
        
        sigma1 = 0.35 * SIZE_PIX
        sigma2 = 1.6 * sigma1
        g1 = cv2.GaussianBlur(image_pt[t], (0,0), sigma1)
        g2 = cv2.GaussianBlur(image_pt[t], (0,0), sigma2)
        dog = g2.astype(np.float32) - g1.astype(np.float32)
    
    df_all = pd.concat(list_df)
    
    print(f'Blob_log {time.time() - top:.1f} s') 
    
    ax = axes[4]
    draw_circles(image_pt[t], blobs, fig = fig, ax = ax)
    
    ax = axes[5]
    ax.imshow(dog)
    
    
    #### LINK
    top = time.time()
    df_all = tp.link(df_all, 4, pos_columns=['y', 'x'], t_column='frame', memory=0, 
                 predictor=None, adaptive_stop=None, adaptive_step=0.95, 
                 neighbor_strategy=None, link_strategy=None, dist_func=None, to_eucl=None)
    print(f'Link {time.time() - top:.1f} s') 
    
    
    
    plt.show()
    out = (blobs, df_all,)
    
    return(out)
    
    # image_raw = skm.io.imread(tifPath)
    # image = skm.util.img_as_float32(image_raw)
    # if EQUALIZE:
    #     for t in range(image.shape[0]):
    #         p1, p99 = np.percentile(image[t].flatten()[mask.flatten()], (1, 99))
    #         image[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
    


# tifPath = "F:\\WorkingData\\26-06-19_FastAcq\\FilmBF_fastAcq_4000f_10Hz_C1.tif"
# tifPath = "C:\\Users\\josep\\Desktop\\Seafile\\AnalysisPulls\\" + \
#           "26-06-19_FastAcq\\FilmBF_fastAcq_4000f_10Hz_C1.tif"
tifPath = "C:\\Users\\josep\\Desktop\\Seafile\\AnalysisPulls\\" + \
          "26-06-10_Test-NileBlueYolk\\M1_40x-WI\\26-06-10_TestNileBlueYolk_C2_10fps_1min_L50p.tif"
dstDir = ""

# out, keypoints = TrackSpotsInCell(tifPath, dstDir)

# %% Pipeline main functions

#### Function
def PretreatImageForTrackMate(tifPath, **kwargs):
    SETTINGS = {
        # 'SCALE' : SCALE_40X,
        # 'SIZE_UM' : 1.5,
        # 'SIZE_PIX' : 1.5/SCALE_40X,
        'EQUALIZE' : False,
        'N_ERODE' : 80,
        'TOP_HAT' : True,
        'MEDIAN_FILTER' : True,
        'SAVE_OUTPUT_IMAGE' : False,
        'RETURN_MASK' : False,
    }
    
    SETTINGS.update(kwargs)
    print(SETTINGS)
    
    # fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    # axes = axes.flatten()
    # iPlot = 0
    
    #### Get image and mask
    shape, dtype = ufun.tiff_inspect(tifPath)
    nT = shape[0]
    
    nT_subset = min(100, nT)
    subset_T = np.linspace(0, nT-1, num = nT_subset, dtype=int)
    image_subset = ufun.load_stack_region(tifPath, time_indices=subset_T, 
                                          x_slice=None, y_slice=None)
    image_subset = skm.util.img_as_float32(image_subset)
    
    inner_cell_contour = get_reasonable_inner_cell_contour(image_subset, PLOT = False)
    mask = ufun.contour_to_mask([shape[1], shape[2]], inner_cell_contour)
    mask = ndi.binary_erosion(mask, iterations = SETTINGS['N_ERODE'])
    
    image = ufun.load_stack_region(tifPath, time_indices=None, 
                                   x_slice=None, y_slice=None)
    # image = skm.util.img_as_float32(image)  
    # image = skm.util.img_as_ubyte(image)
    
    # ax = axes[iPlot]
    # ax.imshow(image[0], cmap='gray')
    # iPlot += 1
    
    #### Pretreatments
    image_pt = np.zeros_like(image)
    
    #### i. EQUALIZE
    if SETTINGS['EQUALIZE']:
        top = time.time()
        for t in range(nT):
            p1, p99 = np.percentile(image[t].flatten()[mask.flatten()], (1, 99))
            image_pt[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
        print(f'Equalize {time.time() - top:.1f} s')    
        
    else:
        for t in range(nT):
            image_pt[t] = image[t, :, :]
    
    #### ii. CROP
    for t in range(nT):
        image_pt[t] = image_pt[t] * mask
            
    # ax = axes[iPlot]
    # ax.imshow(image_pt[0], cmap='gray')
    # iPlot += 1
    
    #### iii. FILTER
    if SETTINGS['TOP_HAT']: # Applying the Top-Hat operation
        top = time.time()
        filterSize = (15, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        for t in range(nT):
            image_pt[t] = cv2.morphologyEx(image_pt[t], cv2.MORPH_TOPHAT, kernel)
            p1, p99 = np.percentile(image_pt[t].flatten()[mask.flatten()], (1, 99))
            image_pt[t] = skm.exposure.rescale_intensity(image_pt[t], in_range=(p1, p99))
        print(f'Top hat {time.time() - top:.1f} s') 
        
    # ax = axes[iPlot]
    # ax.imshow(image_pt[0], cmap='gray')
    # iPlot += 1
    
    if SETTINGS['MEDIAN_FILTER']:
        top = time.time()
        for t in range(nT):
            k = 3
            image_pt[t] = cv2.medianBlur(image_pt[t], k)
        print(f'Median filter {time.time() - top:.1f} s') 
            
    # ax = axes[iPlot]
    # ax.imshow(image_pt[0], cmap='gray')
    # iPlot += 1
    
    # image_to_save = image_pt[:nT]
    # skm.io.imsave(dstDir + pretreatedName, image_to_save)
    # plt.show()
    
    image_output = skm.util.img_as_ubyte(image_pt)
    
    if SETTINGS['SAVE_OUTPUT_IMAGE']:
        srcDir, tifName = os.path.split(tifPath)
        tifRoot = tifName.split('.')[0]
        outputName = tifRoot + '_pretreated.tif'
        outputPath = os.path.join(srcDir, outputName)
        skm.io.imsave(outputPath, image_output)
    
    output = (image_output, )
    
    if SETTINGS['RETURN_MASK']:
        output += (mask, )
    
    if len(output) == 1:
        output == output[0]
        
    return(output)



def runTrackMate(tif_file, xmlPath):
    imp = ij.py.to_imageplus(tif_file)
    dims = imp.getDimensions() # default order: XYCZT
    print(dims)
    
    if dims[4] == 1:
        print('need to change order')
        imp.setDimensions(dims[4], dims[3], dims[2])
    
    
    print(f" dims: {tif_file.dims if hasattr(tif_file, 'dims') else 'N/A'}")
    
    File = sj.jimport('java.io.File')
    Model = sj.jimport('fiji.plugin.trackmate.Model')
    Settings = sj.jimport('fiji.plugin.trackmate.Settings')
    TrackMate = sj.jimport('fiji.plugin.trackmate.TrackMate')
    FeatureFilter = sj.jimport('fiji.plugin.trackmate.features.FeatureFilter')
    LAPUtils = sj.jimport('fiji.plugin.trackmate.tracking.jaqaman.LAPUtils')
    # SelectionModel = sj.jimport('fiji.plugin.trackmate.SelectionModel')
    Logger = sj.jimport('fiji.plugin.trackmate.Logger')
    # DisplaySettingsIO = sj.jimport('fiji.plugin.trackmate.gui.displaysettings.DisplaySettingsIO')
    # HyperStackDisplayer = sj.jimport('fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer')
    
    LogDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.LogDetectorFactory')
    # DogDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.DogDetectorFactory')
    SparseLAPTrackerFactory = sj.jimport('fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory')
    
    TrackAnalyzerProvider = sj.jimport('fiji.plugin.trackmate.providers.TrackAnalyzerProvider')
    FeatureFilter = sj.jimport('fiji.plugin.trackmate.features.FeatureFilter')
    
    # TmXmlWriter = sj.jimport('fiji.plugin.trackmate.io.TmXmlWriter')
    # CSVExporter = sj.jimport('fiji.plugin.trackmate.io.CSVExporter')
    # TrackTableView = sj.jimport('fiji.plugin.trackmate.visualization.table.TrackTableView')
    ExportTracksToXML = sj.jimport('fiji.plugin.trackmate.action.ExportTracksToXML')
    
    # from fiji.plugin.trackmate.io import TmXmlWriter
    # from fiji.plugin.trackmate.io import CSVExporter
    # from fiji.plugin.trackmate.visualization.table import TrackTableView
    # from fiji.plugin.trackmate.action import ExportTracksToXML
    
    
    # Initiate
    model = Model()
    # model.setLogger(Logger.IJ_LOGGER)
    model.setLogger(Logger.DEFAULT_LOGGER)
    
    settings = Settings(imp)
    
    # Configure detector
    settings.detectorFactory = LogDetectorFactory()
    settings.detectorSettings = {
        'DO_SUBPIXEL_LOCALIZATION' : True,
        'RADIUS' : 6.0,
        'TARGET_CHANNEL': ij.py.to_java(1),
        'DO_MEDIAN_FILTERING': False,
        'THRESHOLD': 1.0 # 0.01
    }
    
    # Configure tracker
    settings.trackerFactory = SparseLAPTrackerFactory()
    settings.trackerSettings = LAPUtils.getDefaultSegmentSettingsMap()
    settings.trackerSettings['LINKING_MAX_DISTANCE'] = 3.0
    settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 3.0
    settings.trackerSettings['MAX_FRAME_GAP'] = ij.py.to_java(0)
    
    # Configure filtering
    
    # settings.addAllAnalyzers()
    trackAnalyzerProvider = TrackAnalyzerProvider()
    for key in trackAnalyzerProvider.getKeys():
        print(key)
        settings.addTrackAnalyzer(trackAnalyzerProvider.getFactory(key))
    
    filter1 = FeatureFilter('TRACK_DURATION', 40, True)
    settings.addTrackFilter(filter1)
    
    # Run the model
    trackmate = TrackMate(model, settings)
    ok = trackmate.checkInput()
    if not ok:
        sys.exit(str(trackmate.getErrorMessage()))
    
    ok = trackmate.process()
    if not ok:
        sys.exit(str(trackmate.getErrorMessage()))
    
    model.getLogger().log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks.')
    
    simple_xml_file = File(xmlPath)
    ExportTracksToXML.export(model, settings, simple_xml_file)
    
    print('\nDone!')


# runTrackMate(imp, xmlPath)

# %% Function of the whole pipeline + test

def pretreatAndTrack(tifPath, dstDir):
    srcDir, tifName = os.path.split(tifPath)
    xmlName = tifName.split('.')[0] + '_PyTracks.xml'
    xmlPath = os.path.join(srcDir, xmlName)
    PtImage, mask = PretreatImageForTrackMate(tifPath, 
                                        N_ERODE = 50,
                                        SAVE_OUTPUT_IMAGE = True,
                                        RETURN_MASK = True)
    
    
    
    tif_file = ij.py.to_java(PtImage)
    # tif_file = ij.io().open(srcDir + tifPtName)
    runTrackMate(tif_file, xmlPath)
    
    Tracks = importTrackMateTracks(xmlPath)
    
    I0 = ufun.load_stack_region(tifPath, time_indices=[0])[0]
    Co = ufun.mask_to_contour(mask, keep_only_longest_contour = True)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax = axes[0]
    ax.imshow(I0, cmap='gray')
    
    ax = axes[1]
    ax.imshow(PtImage[0], cmap='gray')
    
    ax = axes[2]
    ax.imshow(I0, cmap='gray')
    ax.plot(Co[:,1], Co[:,0], ls='-', color='darkorange', lw=1.5)
    CL = pm.cL_Set21
    for k in range(len(Tracks)):
        track = Tracks[k]
        color = CL[k%len(CL)]
        ax.plot(track[:,1], track[:,2], ls='-', color=color, lw=0.25)
    
    
    plt.show()
    
    
    return(Tracks)
    

    
#### Arguments

# tifPath = "F:\\WorkingData\\26-06-19_FastAcq\\FilmBF_fastAcq_4000f_10Hz_C1.tif"
# tifPath = "C:\\Users\\josep\\Desktop\\Seafile\\AnalysisPulls\\" + \
#           "26-06-19_FastAcq\\FilmBF_fastAcq_4000f_10Hz_C1.tif"
srcDir = "C:\\Users\\josep\\Desktop\\Seafile\\AnalysisPulls\\" + \
          "26-06-10_Test-NileBlueYolk\\M1_40x-WI\\"
dstDir = srcDir

# tifName = "26-06-10_TestNileBlueYolk_C2_10fps_1min_L50p.tif"
tifName = "26-06-10_TestNileBlueYolk_C1_4fps_5min_L20p.tif"
tifPath = os.path.join(srcDir, tifName)   

xmlName = tifName.split('.')[0] + '_PyTracks.xml'
xmlPath = os.path.join(srcDir, xmlName)
    
# PtImage = PretreatImageForTrackMate(tifPath)
# tif_file = ij.py.to_java(PtImage)

pretreatAndTrack(tifPath, dstDir)
Tracks = importTrackMateTracks(xmlPath)

# %% Analyse tracks

SCALE = SCALE_40X
FPS = 4

Tracks = importTrackMateTracks(xmlPath)
Np = len(Tracks)
    
column_names = ['frame', 'x', 'y', 'particle']
all_tracks = []
for i, track in enumerate(Tracks):
    nT = len(track)
    if nT >= 30:
        track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
        track[:,0] = track[:,0].astype(int) + 1
        all_tracks.append(track)

concat_tracks = np.concat(all_tracks, axis = 0)
df = pd.DataFrame({column_names[k] : concat_tracks[:,k] for k in range(len(column_names))})

# %%%
#### Run msd
res_emsd = tp.motion.emsd(df, SCALE, FPS, max_lagtime=40).reset_index()
T, MSD = res_emsd['lagt'], res_emsd['msd']

parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
D = parms.values[0]/4

parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
b, a = parms
k_full = a
D_full = np.exp(b)/4

print(k_full, D_full)

parms, results = ufun.fitLineHuber(np.log(T[:4]), np.log(MSD[:4]), with_intercept = True)
b, a = parms
k_fast = a
D_fast = np.exp(b)/4

print(k_fast, D_fast)

parms, results = ufun.fitLineHuber(np.log(T[-15:]), np.log(MSD[-15:]), with_intercept = True)
b, a = parms
k_slow = a
D_slow = np.exp(b)/4

print(k_slow, D_slow)

Tc = (D_fast/D_slow)**(1/(k_slow-k_fast))

# %%%

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_xscale('log')
ax.set_yscale('log')

Xp = np.array([1e-3, 1e3])

ax.plot(T, MSD, 'wo', mec='k')
# ax.plot(Xp, 4*D_full*(Xp**k_full), ls='-', color=pm.cL_Set21[0], mec='k', label='Full curve')
ax.plot(Xp, 4*D_fast*(Xp**k_fast), ls='-', color=pm.cL_Set21[1], mec='k', 
        label=f'First 4 pts\n$\\alpha$={k_fast:.2f}')
ax.plot(Xp, 4*D_slow*(Xp**k_slow), ls='-', color=pm.cL_Set21[2], mec='k', 
        label=f'Last 15 pts\n$\\alpha$={k_slow:.2f}')
ax.axvline(Tc, color='gray', lw=1.5, label=f'$T_c$ = {Tc:.2f}')
ax.legend()
ax.grid()
ax.set_xlim([0.5e-1, 2e1])
ax.set_ylim([0.5e-2, 2e0])
plt.show()

# %%%
#### Run imsd -> Might be useful for SEM computation
res_imsd = tp.motion.imsd(df, SCALE, FPS, max_lagtime=30).reset_index()

# %%%

fig, ax = plt.subplots(1, 1)
ax.set_xscale('log')
ax.set_yscale('log')
for p in range(1, Np+1):
    T, MSD = res_imsd['lag time [s]'].values, res_imsd[p].values
    ax.plot(T, MSD, lw=0.5)
ax.set_xlabel('$\\Delta t$ (s)')
ax.set_ylabel('MSD ($\\mu m^2$)')
plt.show()

list_D = []
list_k = []

for p in range(1, Np+1):
    T, MSD = res_imsd['lag time [s]'].values, res_imsd[p].values
    parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
    b, a = parms
    k_nl = a
    D_nl = np.exp(b)/4
    list_k.append(k_nl)
    list_D.append(D_nl)


# for p in range(1, Np+1):
#     T, MSD = res_imsd['lag time [s]'].values[-10:], res_imsd[p].values[-10:]
#     parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
#     b, a = parms
#     k_nl = a
#     D_nl = np.exp(b)/4
#     if D_nl < 0.2 and D_nl > 0 and k_nl > 0:
#        list_k.append(k_nl)
#        list_D.append(D_nl)
    
# %%%

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ax = axes[0]
# sns.swarmplot(ax=ax, y=list_D)
ax.hist(list_D, bins=60, color='gray', zorder=3)
ax.set_xlim([-0.005, 0.18])
ax.set_xlabel('D ($\\mu m^2/s$)')
ax.set_ylabel('N')

ax = axes[1]
# sns.swarmplot(ax=ax, y=list_k)
ax.hist(list_k, bins=60, color='gray', zorder=3)
ax.set_xlabel('exponent $\\alpha$')
ax.set_ylabel('N')

ax = axes[2]
sns.scatterplot(ax=ax, x=list_k, y=list_D, alpha=0.1)
ax.set_xlabel('exponent $\\alpha$')
ax.set_ylabel('D ($\\mu m^2/s$)')

plt.show()
