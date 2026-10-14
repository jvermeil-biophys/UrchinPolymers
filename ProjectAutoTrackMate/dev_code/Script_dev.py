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
# import trackpy as tp
import skimage as skm
# import seaborn as sns
import matplotlib as mpl
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# from shapely.geometry import MultiPoint

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.CalibrationData as cd
import Libs.UtilityFunctions as ufun
# import Libs.ToolboxCytoplasmAnalysis as tbca
# import Libs.ToolboxStructureAnalysis as tbsa



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



def make_NbYolkPronucleus_contour_and_mask(img, PixPerUm,
                                           Contour_cell, Mask_cell,
                                           mode = 'dark_background', 
                                           buffer_pix = 3,
                                           PLOT = False):

    Contour_inner = Contour_cell
    Mask_inner = Mask_cell
    
    # ---
    
    # k_th = 0.9
    nT, nY, nX = img.shape
    th_fun = skm.filters.threshold_otsu
    Mask_inner_f = Mask_inner.flatten()
    
    step = nT//10
    img_masked = img[::step] * Mask_inner
    img_bin = np.zeros_like(img_masked)
    
    if mode == 'dark_background':
        for k, im in enumerate(img_masked):
            im_f = im.flatten()
            th1 = th_fun(im_f[Mask_inner_f]) # * k_th
            img_bin[k] = (im < th1)
            img_bin[k] = ndi.binary_opening(img_bin[k], iterations = 2)
    elif mode == 'bright_background':
        for k, im in enumerate(img_masked):
            im_f = im.flatten()
            th1 = th_fun(im_f[Mask_inner_f]) # * k_th
            img_bin[k] = (im > th1)
            img_bin[k] = ndi.binary_opening(img_bin[k], iterations = 2)
    
    img_bin = np.median(img_bin, axis = 0)
    img_bin = ndi.binary_opening(img_bin, iterations = 10)
    img_bin = ndi.binary_dilation(img_bin, iterations = 1)
    
    img_label, num_features = ndi.label(img_bin)
    properties = ['label', 'area', 'perimeter']
    df = pd.DataFrame(skm.measure.regionprops_table(img_label, img_bin, properties = properties))
    df['circularity'] = 4*np.pi*df['area']/(df['perimeter']**2)
    df = df[df['area'] > 6000]
    df = df[df['area'] < 15000]
    df = df[df['circularity'] > 0.55]

    if len(df) > 0:
        df = df.sort_values(by='area', ascending=False)
        best_label = df['label'].to_numpy()[0]
        mask_pronuc = (img_label == best_label)
        mask_pronuc = ndi.binary_dilation(mask_pronuc, iterations = buffer_pix)
        contour_pronuc = skm.measure.find_contours(mask_pronuc, 0.5)[0]
    else:
        mask_pronuc = None
        contour_pronuc = None
        

    if PLOT:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        axes_f = axes.flatten()
        
        ax = axes_f[0]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(img[0], cmap='gray')
        if not (contour_pronuc is None):
            ax.plot(contour_pronuc[:,1], contour_pronuc[:,0], 'r-')
        
        ax = axes_f[1]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(img_bin, cmap='gray')
        
        if not (mask_pronuc is None):
            ax = axes_f[2]
            ax.set_aspect('equal', adjustable='box')
            ax.imshow(mask_pronuc, cmap='gray')
        
        plt.show()
    
    return contour_pronuc, mask_pronuc



def update_masks(yx, L, total_mask, allowed_mask, NyNx, 
                 force_distribution = False):
    Ny, Nx = NyNx
    y, x = yx
    y1, x1 = y+L, x+L   
    y0, x0 = y-L, x-L
    total_mask[y:y1, x:x1] = False
    
    y1, x1 = min(y1, Ny), min(x1, Nx)
    y0, x0 = max(y0, 0), max(x0, 0)
    allowed_mask[y0:y1, x0:x1] = False
    
    if force_distribution:
        allowed_mask[int(y-L/2):int(y+L/2), :] = False
        allowed_mask[:, int(x-L/2):int(x+L/2)] = False
    
    return(total_mask, allowed_mask)


def get_N_random_square_ROIs_in_cell(mask, L, N, 
                                     excluded_mask=None,
                                     force_distribution = False,
                                     seed=None, PLOT = False):
    Ny, Nx = mask.shape
    
    if not (excluded_mask is None):
        total_mask = mask & (~excluded_mask)
    else:
        total_mask = mask
    
    kernel = np.zeros((L*2 - 1, L*2 - 1), dtype=bool)
    kernel[L:, L:] = True
    
    allowed_mask = ndi.binary_erosion(total_mask, structure=kernel, iterations=1)
    # allowed_mask = cv2.erode(total_mask, kernel, iterations=1)
    
    if PLOT:
        fig, axes = plt.subplots(1, N+1, figsize=((N+1)*3, 3.5))
        ax = axes[0]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(total_mask*0.5 + allowed_mask*0.5, 
                  cmap='gray', vmin=0, vmax=1)
    
    list_rois = []
    
    for k in range(N):
    
        coords = np.argwhere(allowed_mask)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, coords.shape[0], 1, dtype=int)
        
        y0, x0 = coords[idx,:][0]
        y1, x1 = y0+L, x0+L
        list_rois.append([[int(y0), int(x0)], 
                          [int(y1), int(x1)]])
        
        # total_mask[y0:y1, x0:x1] = False
        # erroded_mask = ndi.binary_erosion(total_mask, structure=kernel, iterations=1)
        
        total_mask, allowed_mask = update_masks((y0, x0), L, 
                                                total_mask, allowed_mask, (Ny, Nx),
                                                force_distribution = force_distribution)
    
        if PLOT:
            ax = axes[k+1]
            ax.set_aspect('equal', adjustable='box')
            ax.imshow(total_mask*0.5 + allowed_mask*0.5, 
                      cmap='gray', vmin=0, vmax=1)
            for roi in list_rois:
                y0, x0 = roi[0]
                ax.plot([x0-0.5], [y0-0.5], 'ro', markersize=2)
                rect = mpl.patches.Rectangle((x0-0.5, y0-0.5), L, L,
                                             facecolor='None', edgecolor='g')
                ax.add_patch(rect)

    if PLOT:
        plt.show()
    
    return(list_rois)


# %%% Script









# %% 2. Run TrackMate

# %%% Imports 2

os.environ["JAVA_HOME"] = up.Path_JAVA_HOME

import imagej
import scyjava as sj

sj.config.add_options('-Xmx15g') # Digits are the amount of RAM assigned

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


def export_as_TrackMate_tracks(
        tracks,
        filepath,
        spaceUnits="pixels",
        frameInterval=1.0,
        timeUnits="frames",
        generationDateTime=None,
        from_version="TrackMate v7.14.0"
    ):
    """
    Export a list of particle tracks to a TrackMate-style XML file.

    Parameters
    ----------
    tracks : list of numpy.ndarray
        Each array has shape (N, 3), with columns [t, x, y].
    filepath : str
        Output XML file path.
    spaceUnits : str
        Spatial units, e.g. "pixels" or "microns".
    frameInterval : float
        Time interval between frames.
    timeUnits : str
        Time units, e.g. "frames" or "seconds".
    generationDateTime : str or None
        Date/time string for the XML header.
    from_version : str
        TrackMate version string for the XML header.
    """

    if generationDateTime is None:
        generationDateTime = ""

    # Root element
    root = ET.Element(
        "Tracks",
        {
            "nTracks": str(len(tracks)),
            "spaceUnits": str(spaceUnits),
            "frameInterval": str(frameInterval),
            "timeUnits": str(timeUnits),
            "generationDateTime": str(generationDateTime),
            "from": str(from_version),
        }
    )

    # Add each particle and its detections
    for track in tracks:
        track = np.asarray(track)

        if track.size == 0:
            particle = ET.SubElement(
                root, "particle", {"nSpots": "0"}
            )
            continue

        if track.ndim != 2 or track.shape[1] != 3:
            raise ValueError(
                "Each track must be a 2D array with columns [t, x, y]."
            )

        particle = ET.SubElement(
            root,
            "particle",
            {"nSpots": str(len(track))}
        )

        for t, x, y in track:
            ET.SubElement(
                particle,
                "detection",
                {
                    "t": str(t),
                    "x": str(x),
                    "y": str(y),
                    "z": "0.0",
                }
            )

    # Pretty-print the XML with indentation
    ET.indent(root, space="  ")

    # Write XML to disk
    tree = ET.ElementTree(root)
    tree.write(
        filepath,
        encoding="UTF-8",
        xml_declaration=True
    )


def run_TrackMate(imp, xmlPath, PixPerUm,  # tif_file
                 ROI = None,
                 RADIUS_UM = 1.0, 
                 THRESH_SPOT_QLT = 1.0,
                 THRESH_LINK_UM = 0.2, 
                 THRESH_MIN_DURATION = 40,
                 ):
    
    # imp = ij.py.to_imageplus(tif_file)
    dims = imp.getDimensions() # default order: XYCZT
    print(dims)
    
    if dims[4] == 1:
        print('need to change order')
        imp.setDimensions(dims[4], dims[3], dims[2])
        
    dims = imp.getDimensions() # default order: XYCZT
    # print(dims)
    
    
    # print(f" dims: {tif_file.dims if hasattr(tif_file, 'dims') else 'N/A'}")
    
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
    
    # SettingsClass = settings.getClass()
    # for method in SettingsClass.getMethods():
    #     name = str(method.getName()).lower()
    #     if 'roi' in name or 'region' in name or 'area' in name:
    #         print(method)
    
    # print(sj.jimport('fiji.plugin.trackmate.TrackMate').PLUGIN_NAME_VERSION)
    # print(settings.getClass().getField('roi'))
    
    RADIUS = PixPerUm * RADIUS_UM
    THRESH_LINK = PixPerUm * THRESH_LINK_UM
    
    # if ROI is not None:
    #     Roi = sj.jimport('ij.gui.Roi')
    #     x0, y0 = ROI[0]
    #     x1, y1 = ROI[1]
    #     x = min(x0, x1)
    #     y = min(y0, y1)
    #     width = abs(x1 - x0)
    #     height = abs(y1 - y0)
    #     # settings.roi = Roi(
    #     #     int(x), int(y), int(width), int(height)
    #     # )
    #     settings.setRoi(Roi(int(x), int(y), int(width), int(height)))
        
    
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
    

def pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                              Dict_TrackMate_Settings = None,
                              ROI = None, correct_on_edges = False,
                              Mask_cell = None, mask_buffer_um = 0.0, 
                              PLOT = False, SHOWPLOT = False, SAVEPLOT = False):
    
    srcDir, tifName = os.path.split(tifPath)
    rawTrackPath = os.path.join(dstDir, rawTrackName)
    
    shape, dtype = ufun.tiff_inspect(tifPath)
    nT = shape[0]
    
    if ROI is not None:
        x0, y0 = ROI[0]
        x1, y1 = ROI[1]
        xroi = min(x0, x1)
        yroi = min(y0, y1)
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        x_slice=slice(int(xroi), int(xroi + width))
        y_slice=slice(int(yroi), int(yroi + height))
        
    else:
        x_slice=None
        y_slice=None    
    
    image = ufun.load_stack_region(tifPath, time_indices=None, 
                                   x_slice=x_slice, y_slice=y_slice)
    
    # nT = 400
    # image = ufun.load_stack_region(tifPath, time_indices=range(0, nT), 
    #                                x_slice=x_slice, y_slice=y_slice)
    
    
    if Mask_cell is None:
        pass
    else:
        mask_buffer_pix = round(mask_buffer_um * PixPerUm)
        Mask_cell = ndi.binary_erosion(Mask_cell, iterations=mask_buffer_pix)
        image = image * Mask_cell
    
    
    # Pretreatments
    for t in range(nT):
        k = 3
        image[t] = cv2.medianBlur(image[t], k)
    
    image_0 = image[0,:,:]
    # tif_file = ij.py.to_java(image)
    # imp = ij.py.to_imageplus(tif_file)

    imp = ij.py.to_imageplus(image)
    del(image)
      
    # Update TrackMate settings
    Dict_TrackMate_Settings_DEFAULTS = {
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
    # runTrackMate(tif_file, rawTrackPath, PixPerUm, ROI = ROI,
    #              **Dict_TrackMate_Settings)
    run_TrackMate(imp, rawTrackPath, PixPerUm, ROI = ROI,
                  **Dict_TrackMate_Settings)
    
    
    # Open the tracks and do the post-treatment
    Tracks = import_TrackMate_tracks(rawTrackPath)
    print('len Tracks', len(Tracks))
    if correct_on_edges:
        # valid = np.ones(len(Tracks), dtype=bool)
        Valid_Tracks = []
        for k, track in enumerate(Tracks):
            xm = np.median(track[:,1])
            ym = np.median(track[:,2])
            dist_x = min(np.abs(xm-0), np.abs(xm-width))
            dist_y = min(np.abs(ym-0), np.abs(ym-height))
            if not ((dist_x < 0.5*PixPerUm) or (dist_y < 0.5*PixPerUm)):
                Valid_Tracks.append(track)
                # print(f'ok : {k:.0f}')
            # else:
                # print(dist_x, dist_y)
                
        Tracks = Valid_Tracks
    print('len Tracks', len(Tracks))
        
    if ROI is not None:
        for track in Tracks:
            track[:, 1] += xroi
            track[:, 2] += yroi
            
    export_as_TrackMate_tracks(Tracks, rawTrackPath)
            
    
    # Plot
    if PLOT:
        if not SHOWPLOT:
            plt.ioff()
        else:
            plt.ion()
            
        pm.setGraphicOptions(mode = 'screen')
        image_raw_0 = ufun.load_stack_region(tifPath, time_indices=[0])[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('_'.join(tifName.split('_')[:5]))
        
        ax = axes[0]
        ax.imshow(image_raw_0, cmap='gray')
        if ROI is not None:
            rect = mpl.patches.Rectangle((xroi-0.5, yroi-0.5), width, height,
                                         facecolor='None', edgecolor='r')
            ax.add_patch(rect)
        
        ax = axes[1]
        ax.imshow(image_0, cmap='gray')
        
        ax = axes[2]
        ax.imshow(image_raw_0, cmap='gray')
        if ROI is not None:
            rect = mpl.patches.Rectangle((xroi-0.5, yroi-0.5), width, height,
                                         facecolor='None', edgecolor='r')
            ax.add_patch(rect)
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
    return(Tracks)
    

  

# %%% Script


mainDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D2')
dstDir = os.path.join(mainDir, 'SPT_results')
tifNames = [
    '26-09-30_D2-R_PreF_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PreF_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    ]


tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]


UmPerPix = cd.UmPerPix_60X_W1
PixPerUm = 1/UmPerPix
Dict_Image_Settings = {
    'UmPerPix' : UmPerPix,
    'PixPerUm' : 1/UmPerPix,
    'FPS' : 20,
    }

i = 0

tifPath, fN = tifPaths[i], fileNames[i]
fileROIs = fN + '_listSquareROIs'

shape, dtype = ufun.tiff_inspect(tifPath)
nT = shape[0]
TT = range(0, nT, nT//100)
img = ufun.load_stack_region(tifPath, time_indices=TT)

top = time.time()
print(0)

Contour_cell, Mask_cell = make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                                           buffer_um = 0.0,
                                                           mode = 'dark_background', 
                                                           PLOT = True)

print(1)
print(f'{time.time() - top:.3f} s')

Contour_pronuc, Mask_pronuc = make_NbYolkPronucleus_contour_and_mask(img, PixPerUm,
                                                               Contour_cell, Mask_cell,
                                                               mode = 'dark_background', 
                                                               buffer_pix = 3,
                                                               PLOT = True)

print(2)
print(f'{time.time() - top:.3f} s')

L = int(15/UmPerPix)
N = 3

list_rois = get_N_random_square_ROIs_in_cell(Mask_cell, L, N, 
                                             excluded_mask = Mask_pronuc,
                                             force_distribution = True,
                                             seed=4, PLOT = True)


ufun.list2json(list_rois, srcDir, fileROIs)

print(3)
print(f'{time.time() - top:.3f} s')


# %% Re-run from here

top = time.time()
list_rois = ufun.json2list(srcDir, fileROIs)

Dict_TrackMate_Settings = {
    'RADIUS_UM' : 0.8, 
    'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 
    'THRESH_MIN_DURATION' : 30,
    }

for k, ROI in enumerate(list_rois[:1]):
    rawTrackName = fN + f'_sqROI-0{k:.0f}.xml'
    
    print('\n----\n')
    print(4+k)
    
    pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, ROI = ROI,
                              Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                              correct_on_edges = True,
                              Mask_cell = None, mask_buffer_um = 0.0,
                              PLOT = True, SHOWPLOT = True, SAVEPLOT = False)
    
    # runTrackMate(tifPath, xmlPath, PixPerUm,
    #              RADIUS_UM = 1.0, 
    #              THRESH_SPOT_QLT = 1.0,
    #              THRESH_LINK_UM = 0.2, 
    #              THRESH_MIN_DURATION = 40,
    #              ROI = ROI)
    
    
    print(f'{time.time() - top:.3f} s')







# %% Tests 

# %%%


def runTrackMate(tif_file, xmlPath, PixPerUm,
                 RADIUS_UM = 1.0, 
                 THRESH_SPOT_QLT = 1.0,
                 THRESH_LINK_UM = 0.2, 
                 THRESH_MIN_DURATION = 40):
    
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
    Logger = sj.jimport('fiji.plugin.trackmate.Logger')
    
    LogDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.LogDetectorFactory')
    SparseLAPTrackerFactory = sj.jimport('fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory')
    
    TrackAnalyzerProvider = sj.jimport('fiji.plugin.trackmate.providers.TrackAnalyzerProvider')
    FeatureFilter = sj.jimport('fiji.plugin.trackmate.features.FeatureFilter')
    
    ExportTracksToXML = sj.jimport('fiji.plugin.trackmate.action.ExportTracksToXML')
    
    
    # Initiate
    model = Model()
    model.setLogger(Logger.DEFAULT_LOGGER)
    
    settings = Settings(imp)
    
    RADIUS = PixPerUm * RADIUS_UM
    THRESH_LINK = PixPerUm * THRESH_LINK_UM
    
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


# %%% Script -- Test of make_NbYolkPronucleus_contour_and_mask()

# %%%% Test of the script

mainDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D1')
dstDir = os.path.join(mainDir, 'SPT_results')
tifNames = [
    '26-09-30_D1_PreF_C1_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PreF_C2_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PreF_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    ]

mainDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D2')
dstDir = os.path.join(mainDir, 'SPT_results')
tifNames = [
    '26-09-30_D2-R_PreF_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PreF_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    ]


tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

UmPerPix = cd.UmPerPix_60X_W1
PixPerUm = 1/UmPerPix
Dict_Image_Settings = {
    'UmPerPix' : UmPerPix,
    'PixPerUm' : 1/UmPerPix,
    'FPS' : 20,
    }

i = 0

tP, fN = tifPaths[i], fileNames[i]

shape, dtype = ufun.tiff_inspect(tP)
nT = shape[0]
TT = range(0, nT, nT//100)
img = ufun.load_stack_region(tP, time_indices=TT)

top = time.time()
print(0)

Contour_cell, Mask_cell = make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                                           buffer_um = 0.0,
                                                           mode = 'dark_background', 
                                                           PLOT = True)

print(1)
print(f'{time.time() - top:.3f} s')

Contour_pronuc, Mask_pronuc = make_NbYolkPronucleus_contour_and_mask(img, PixPerUm,
                                                               Contour_cell, Mask_cell,
                                                               mode = 'dark_background', 
                                                               buffer_pix = 3,
                                                               PLOT = True)

print(2)
print(f'{time.time() - top:.3f} s')

L = int(15/UmPerPix)
N = 3

list_rois = get_N_random_square_ROIs_in_cell(Mask_cell, L, N, 
                                             excluded_mask = Mask_pronuc,
                                             force_distribution = True,
                                             seed=4, PLOT = True)

print(3)
print(f'{time.time() - top:.3f} s')


# %%%% Test of the function

# def make_NbYolkPronucleus_contour_and_mask(img, PixPerUm,
#                                            Contour_cell, Mask_cell,
#                                            mode = 'dark_background', 
#                                            buffer_pix = 3,
#                                            PLOT = False):

Contour_inner = Contour_cell
Mask_inner = Mask_cell
PixPerUm = 1/cd.UmPerPix_60X_W1
mode = 'dark_background'
buffer_pix = 3
PLOT = True

# ---

# k_th = 0.9
nT, nY, nX = img.shape
th_fun = skm.filters.threshold_otsu
Mask_inner_f = Mask_inner.flatten()

step = nT//10
img_masked = img[::step] * Mask_inner
img_bin = np.zeros_like(img_masked)

if mode == 'dark_background':
    for k, im in enumerate(img_masked):
        im_f = im.flatten()
        th1 = th_fun(im_f[Mask_inner_f]) # * k_th
        img_bin[k] = (im < th1)
        img_bin[k] = ndi.binary_opening(img_bin[k], iterations = 2)
elif mode == 'bright_background':
    for k, im in enumerate(img_masked):
        im_f = im.flatten()
        th1 = th_fun(im_f[Mask_inner_f]) # * k_th
        img_bin[k] = (im > th1)
        img_bin[k] = ndi.binary_opening(img_bin[k], iterations = 2)

img_bin = np.median(img_bin, axis = 0)
img_bin = ndi.binary_opening(img_bin, iterations = 10)
img_bin = ndi.binary_dilation(img_bin, iterations = 1)

img_label, num_features = ndi.label(img_bin)
properties = ['label', 'area', 'perimeter']
df = pd.DataFrame(skm.measure.regionprops_table(img_label, img_bin, properties = properties))
df['circularity'] = 4*np.pi*df['area']/(df['perimeter']**2)
df = df[df['area'] > 6000]
df = df[df['area'] < 15000]
df = df[df['circularity'] > 0.55]

if len(df) > 0:
    df = df.sort_values(by='area', ascending=False)
    best_label = df['label'].to_numpy()[0]
    mask_pronuc = (img_label == best_label)
    mask_pronuc = ndi.binary_dilation(mask_pronuc, iterations = buffer_pix)
    contour_pronuc = skm.measure.find_contours(mask_pronuc, 0.5)[0]
else:
    mask_pronuc = None
    contour_pronuc = None
    

if PLOT:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes_f = axes.flatten()
    
    ax = axes_f[0]
    ax.set_aspect('equal', adjustable='box')
    ax.imshow(img[0], cmap='gray')
    if not contour_pronuc == None:
        ax.plot(contour_pronuc[:,1], contour_pronuc[:,0], 'r-')
    
    ax = axes_f[1]
    ax.set_aspect('equal', adjustable='box')
    ax.imshow(img_bin, cmap='gray')
    
    if not mask_pronuc == None:
        ax = axes_f[2]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(mask_pronuc, cmap='gray')
    
    plt.show()

# return contour_pronuc, mask_pronuc


# %%% Script --- Test for get_N_random_square_ROIs_in_cell()

Xc = 300
Yc = 300
Rc = 250
shape = [600, 600]
Angles = np.linspace(0, 2*np.pi, 360)
Xcontour = Xc + Rc*np.cos(Angles)
Ycontour = Yc + Rc*np.sin(Angles)
contour = np.array([Ycontour, Xcontour]).T
mask = ufun.contour_to_mask(shape, contour)


Xc = 410
Yc = 410
Rc = 70
shape = [600, 600]
Angles = np.linspace(0, 2*np.pi, 360)
Xcontour = Xc + Rc*np.cos(Angles)
Ycontour = Yc + Rc*np.sin(Angles)
excluded_contour = np.array([Ycontour, Xcontour]).T
excluded_mask = ufun.contour_to_mask(shape, excluded_contour)

total_mask = mask & (~excluded_mask)

L = 40
kernel = np.zeros((L*2 - 1, L*2 - 1))
kernel[:L,:L] = 1
erroded_mask = ndi.binary_erosion(total_mask, structure=kernel, iterations=1)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

ax = axes[0]
ax.imshow(total_mask, cmap='gray')

ax = axes[1]
ax.imshow(erroded_mask, cmap='gray')

plt.show()

L = 50
N = 5

list_rois = get_N_random_square_ROIs_in_cell(mask, L, N, 
                                             excluded_mask=excluded_mask,
                                             seed = None, PLOT = True)


















