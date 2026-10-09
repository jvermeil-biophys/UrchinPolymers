# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:55:14 2026

@author: Utilisateur
"""

# %% Imports

import os
import time
import alphashape
import cv2
import sys

import numpy as np
import pandas as pd
import trackpy as tp
import skimage as skm
import seaborn as sns
import matplotlib as mpl
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from shapely.geometry import MultiPoint

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.CalibrationData as cd
import Libs.UtilityFunctions as ufun
import Libs.ToolboxCytoplasmAnalysis as tbca
import Libs.ToolboxStructureAnalysis as tbsa



# %% 0. Paths

# %% 1. Define tracking ROIs

# %%% Subfunctions

def make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                     mode = 'dark_background', 
                                     buffer_um = 0,
                                     PLOT = False):
    nT, nY, nX = img.shape
    k_th = 1.0
    
    if mode == 'dark_background':
        img_proj = np.max(img, axis = 0)
        th1 = skm.filters.threshold_li(img_proj) * k_th
        img_bin = (img_proj > th1)
        
    elif mode == 'light_background':
        img_proj = np.min(img, axis = 0)
        th1 = skm.filters.threshold_li(img_proj) * k_th
        img_bin = (img_proj < th1)
    
    img_bin = ndi.binary_opening(img_bin, iterations = 5)
    img_bin = ndi.binary_fill_holes(img_bin)
    
    # Get contours from first mask
    FoundContours = skm.measure.find_contours(img_bin, 0.5)
    
    # Get additionnal contours by "erroding" the mask
    N_it = 2
    MoreContours = FoundContours[:]
    for k in range(N_it):
        img_bin_bis = ndi.binary_erosion(img_bin, iterations=k+1)
        MoreContours += skm.measure.find_contours(img_bin_bis, 0.5)

    # Concatenating all contours from MoreContours gives a "thick" contour
    concat_contours = np.concatenate(MoreContours)
    # points = MultiPoint(concat_contours[:,::-1])
    # x_ch, y_ch = points.convex_hull.exterior.xy
    
    # Run the alpha shape
    # PixPerUm is the scale in Pix Per Um
    # 20x -> 2.2 ; 40x -> 4.5 ; 60x -> 9.2
    # -> One cell is more pixels at 60x than 20x
    
    # ALPHA sets the size of the "smoothing circle"
    # R = 1/ALPHA -> High alpha = high def; Low alpha = crude def
    
    # Need to check with other images
    
    R = PixPerUm # Radius of 1 µm
    ALPHA = 1/R
    alpha_shape = alphashape.alphashape(concat_contours[:,::-1], ALPHA)
    x_as, y_as = alpha_shape.exterior.xy
    Contour_alpha = np.array([y_as, x_as]).T
    
    inner_shape = alpha_shape.buffer(- buffer_um * PixPerUm)
    x_is, y_is = inner_shape.exterior.xy
    Contour_inner = np.array([y_is, x_is]).T
    
    Mask_inner = ufun.contour_to_mask((nY, nX), Contour_inner)
    
    
    if PLOT:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        axes_f = axes.flatten()
        
        ax = axes_f[0]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(img_proj, cmap='gray')
        
        ax = axes_f[1]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(img_bin, cmap='gray')
        for c in FoundContours:
            ax.plot(c[:, 1], c[:, 0], lw=1)
        # ax.plot(x_ch, y_ch, lw=1)
            
        ax = axes_f[2]
        ax.set_aspect('equal', adjustable='box')
        for c in MoreContours:
            ax.plot(c[:, 1], c[:, 0], lw=1)
            
        ax = axes_f[3]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(Mask_inner, cmap='gray')
        
        for i in [0, 1, 2, 3]:
            ax = axes_f[i]
            ax.set_aspect('equal', adjustable='box')
            ax.plot(Contour_alpha[:, 1], Contour_alpha[:, 0], color='red', lw='0.75')
            ax.plot(Contour_inner[:, 1], Contour_inner[:, 0], color='cyan', lw='0.75')
            # ax.plot(x_is, y_is, lw='0.75')
        
        # list_geoms = list(alpha_shape.geoms)
        # for poly in list_geoms:
        #     x_as, y_as = poly.exterior.xy
        #     ax.plot(x_as, y_as, lw='0.75')
        
        plt.show()
        
    return(Contour_inner, Mask_inner)



def get_N_random_square_ROIs_in_cell():
    #### !!!! TBD
    pass






# %%% Script




















# %% 2. Run TrackMate

# %%% Imports 2

os.environ["JAVA_HOME"] = up.Path_JAVA_HOME

import imagej
import scyjava as sj

sj.config.add_options('-Xmx32g') # Digits are the amount of RAM assigned

# initialize ImageJ
ij = imagej.init(up.Path_Fiji, add_legacy=False)

print(f"ImageJ version: {ij.getVersion()}")


# %%% Subfunctions

def import_TrackMate_tracks(filepath):
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


def runTrackMate(tif_file, xmlPath, Pix_Per_Um,
                 RADIUS_UM = 1.0, 
                 THRESH_SPOT_QLT = 1.0,
                 THRESH_LINK_UM = 0.2, 
                 THRESH_MIN_DURATION = 40):
    
    #### !!!! Modify so it can include a selection !
    
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
    
# =============================================================================
#     Geometry:
#   X =  209 -  381, dx = 0,108333
#   Y =  317 -  512, dy = 0,108333
#   Z =    0 -    0, dz = 1,00000
#   T =    0 - 1999, dt = 0,0500075
# =============================================================================
    
    # Initiate
    model = Model()
    # model.setLogger(Logger.IJ_LOGGER)
    model.setLogger(Logger.DEFAULT_LOGGER)
    
    settings = Settings(imp)
    
    RADIUS = Pix_Per_Um * RADIUS_UM
    THRESH_LINK = Pix_Per_Um * THRESH_LINK_UM
    
    # Configure detector
    settings.detectorFactory = LogDetectorFactory()
    settings.detectorSettings = {
        'DO_SUBPIXEL_LOCALIZATION' : True,
        'RADIUS' : RADIUS,
        'TARGET_CHANNEL': ij.py.to_java(1),
        'DO_MEDIAN_FILTERING': False,
        'THRESHOLD': THRESH_SPOT_QLT # 0.01
    }
    
    # Configure tracker   
    settings.trackerFactory = SparseLAPTrackerFactory()
    settings.trackerSettings = LAPUtils.getDefaultSegmentSettingsMap()
    settings.trackerSettings['LINKING_MAX_DISTANCE'] = THRESH_LINK
    settings.trackerSettings['ALLOW_GAP_CLOSING'] = False
    settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 1.0
    settings.trackerSettings['MAX_FRAME_GAP'] = ij.py.to_java(0)
    
    # Configure filtering
    
    # settings.addAllAnalyzers()
    trackAnalyzerProvider = TrackAnalyzerProvider()
    for key in trackAnalyzerProvider.getKeys():
        print(key)
        settings.addTrackAnalyzer(trackAnalyzerProvider.getFactory(key))
    
    filter1 = FeatureFilter('TRACK_DURATION', THRESH_MIN_DURATION, True)
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
    

def pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir,
                              Pix_Per_Um, Dict_TrackMate_Settings = None,
                              Mask_cell = None, mask_buffer_um = 0.0,
                              return_tracks = False,
                              PLOT = False, SHOWPLOT = False, SAVEPLOT = False):
    
    
    
    srcDir, tifName = os.path.split(tifPath)
    rawTrackPath = os.path.join(dstDir, rawTrackName)
    
    shape, dtype = ufun.tiff_inspect(tifPath)
    nT = shape[0]
    
    image = ufun.load_stack_region(tifPath, time_indices=None, 
                                   x_slice=None, y_slice=None)
    
    # nT = 100
    # image = ufun.load_stack_region(tifPath, time_indices=range(0, 100), 
    #                                x_slice=None, y_slice=None)
    
    if Mask_cell is None:
        pass
    else:
        mask_buffer_pix = round(mask_buffer_um * Pix_Per_Um)
        Mask_cell = ndi.binary_erosion(Mask_cell, iterations=mask_buffer_pix)
        image = image * Mask_cell
        
        
    # Pretreatments
    for t in range(nT):
        k = 3
        image[t] = cv2.medianBlur(image[t], k)
    
    image_0 = image[0,:,:]
    tif_file = ij.py.to_java(image)
    del(image)
      
    # Update TrackMate settings
    Dict_TrackMate_Settings_DEFAULTS = {
        'IMG_UNITS' : 'PIX',
        'RADIUS_UM' : 1.0, 
        'THRESH_SPOT_QLT' : 1.0,
        'THRESH_LINK_UM' : 0.2, 
        'THRESH_MIN_DURATION' : 40,
        }
    if Dict_TrackMate_Settings is None:
        Dict_TrackMate_Settings = Dict_TrackMate_Settings_DEFAULTS
    else:
        Dict_TrackMate_Settings_DEFAULTS.update(Dict_TrackMate_Settings)
        Dict_TrackMate_Settings = Dict_TrackMate_Settings_DEFAULTS
    print(pm.GREEN + 'Settings: ' + pm.NORMAL, Dict_TrackMate_Settings)
    
    
    # Run TrackMate
    runTrackMate(tif_file, rawTrackPath, Pix_Per_Um,
                 **Dict_TrackMate_Settings)
    
    
    # Plot
    if PLOT:
        if not SHOWPLOT:
            plt.ioff()
        else:
            plt.ion()
            
        pm.setGraphicOptions(mode = 'screen')
        Tracks = import_TrackMate_tracks(rawTrackPath)
        image_raw_0 = ufun.load_stack_region(tifPath, time_indices=[0])[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('_'.join(tifName.split('_')[:5]))
        
        ax = axes[0]
        ax.imshow(image_raw_0, cmap='gray')
        
        ax = axes[1]
        ax.imshow(image_0, cmap='gray')
        
        ax = axes[2]
        ax.imshow(image_raw_0, cmap='gray')
        CL = pm.cL_Set21
        for k in range(len(Tracks)):
            track = Tracks[k]
            color = CL[k%len(CL)]
            ax.plot(track[:,1], track[:,2], ls='-', color=color, lw=0.25)
        
        if SHOWPLOT:
            plt.show()
        
        if SAVEPLOT:
            figName = tifName.split('.')[0] + '_rawTracks.png'
            figPath = os.path.join(dstDir, figName)
            fig.savefig(figPath, dpi=500, )
    
    if not SHOWPLOT:
        plt.ion()
    
    
    # Return tracks
    if return_tracks:
        if not PLOT:
            Tracks = import_TrackMate_tracks(rawTrackPath)
        else:
            pass
        
        return(Tracks)
    

  

# %%% Script



















