# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:50:16 2021
@authors: Joseph Vermeil

Toolbox_AutoTrackmate.py - DESCRIPTION

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

# %% Imports

import os
import cv2
import sys
import json
import tifffile
import alphashape

import numpy as np
import pandas as pd
import skimage as skm
import seaborn as sns
import matplotlib as mpl
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

from shapely.geometry import Polygon, MultiPoint

# %% Imports 2

Path_Fiji = "C:\\Users\\Utilisateur\\Desktop\\Fiji_stable.app"
Path_JAVA_HOME = "C:\\Users\\Utilisateur\\miniforge3\\envs\\pyimagej-env\\Library\\lib\\jvm"

os.environ["JAVA_HOME"] = Path_JAVA_HOME

import imagej
import scyjava as sj

# initialize ImageJ
sj.config.add_options('-Xmx12g') # Digits are the amount of RAM assigned
ij = imagej.init(Path_Fiji, add_legacy=False)

print(f"ImageJ version: {ij.getVersion()}") # Should return: ImageJ version: 2.16.0/1.54p


# %% Microscope scales

# LEICA
PixPerUm_10X_Leica = 1.1
PixPerUm_20X_Leica = 2.203
PixPerUm_40X_Leica = 4.292
UmPerPix_10X_Leica = 1/PixPerUm_10X_Leica
UmPerPix_20X_Leica = 1/PixPerUm_20X_Leica
UmPerPix_40X_Leica = 1/PixPerUm_40X_Leica

# X1 (NIKON1)
PixPerUm_20X_X1 = 3.7
PixPerUm_40X_X1 = 7.5
PixPerUm_60X_X1 = 11.2
PixPerUm_100X_X1 = 18.3
UmPerPix_20X_X1 = 1/PixPerUm_20X_X1
UmPerPix_40X_X1 = 1/PixPerUm_40X_X1
UmPerPix_60X_X1 = 1/PixPerUm_60X_X1
UmPerPix_100X_X1 = 1/PixPerUm_100X_X1

# NIKON2
PixPerUm_10X_Nikon2 = 1.55
PixPerUm_20X_Nikon2 = 3.08445
PixPerUm_40X_Nikon2 = 6.1
PixPerUm_100X_Nikon2 = 15.3
UmPerPix_10X_Nikon2 = 1/PixPerUm_10X_Nikon2
UmPerPix_20X_Nikon2 = 1/PixPerUm_20X_Nikon2
UmPerPix_40X_Nikon2 = 1/PixPerUm_40X_Nikon2
UmPerPix_100X_Nikon2 = 1/PixPerUm_100X_Nikon2

# W1 (NIKON3)
PixPerUm_20X_W1 = 3.086
PixPerUm_40X_W1 = 6.135
PixPerUm_60X_W1 = 9.26
PixPerUm_100X_W1 = 15.601
UmPerPix_20X_W1 = 1/PixPerUm_20X_W1
UmPerPix_40X_W1 = 1/PixPerUm_40X_W1
UmPerPix_60X_W1 = 1/PixPerUm_60X_W1
UmPerPix_100X_W1 = 1/PixPerUm_100X_W1


# %% Graphic settings

palette_Set2 = sns.color_palette("Set2")
cL_Set2 = palette_Set2.as_hex()

palette_Set1 = sns.color_palette("Set1")
cL_Set1 = palette_Set1.as_hex()[:-1]

cL_Set12 = cL_Set1 + cL_Set2
cL_Set21 = cL_Set2 + cL_Set1

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

def setGraphicOptions(mode = 'screen', palette = 'Set2', colorList = cL_Set21):
    if mode == 'screen':
        SMALLER_SIZE = 10
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16
    if mode == 'screen_big':
        SMALLER_SIZE = 12
        SMALL_SIZE = 14
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
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    
    sns.set_palette(sns.color_palette(palette))
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colorList) 




# %% Base subfunctions


def list2json(L, filePath):
    L = list(L)
    with open(filePath, 'w') as fp:
        json.dump(L, fp, indent=4)
        

def json2list(filePath):
    with open(filePath, 'r') as fp:
        L = json.load(fp)
    L = list(L)
    return(L)


def tiff_inspect(filepath):
    with tifffile.TiffFile(filepath) as tif:
        series = tif.series[0]  # first series
        shape = series.shape
        dtype = series.dtype
    return(shape, dtype)


def load_stack_region(filepath, time_indices=None, x_slice=None, y_slice=None):
    """
    Load a cropped region of a 3D TIFF (X, Y, time) with minimal memory usage.

    Parameters
    ----------
    filename : str
        Path to the TIFF file.
    time_indices : list[int] or slice, optional
        Which time points to load. Default = all.
    x_slice : slice, optional
        Cropping along X dimension (cols).
    y_slice : slice, optional
        Cropping along Y dimension (rows).

    Returns
    -------
    numpy.ndarray
        Cropped stack with shape (T, Y, X).
    """
    
    with tifffile.TiffFile(filepath) as tif:
        series = tif.series[0]   # the first image series
        pages = series.pages
        firstFrame = pages[0]
        if time_indices is None:
            time_indices = range(0, len(pages))
        if x_slice is None:
            x_slice = slice(0, firstFrame.shape[1])
        if y_slice is None:
            y_slice = slice(0, firstFrame.shape[0])

        # Collect requested frames without loading everything
        cropped_stack = []
        for i in time_indices:
            page = pages[i]
            arr = page.asarray()[y_slice, x_slice]  # crop directly
            cropped_stack.append(arr)

        return(np.stack(cropped_stack, axis=0))


def contour_to_mask(shape, contour):
    """
    Adapted from https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
    """
    nY, nX = shape
    poly = np.round(contour[:,::-1], 0).astype(int).tolist()
    poly = [(p[0], p[1]) for p in poly[:]] + [(poly[0][0], poly[0][1])]
    Im0 = Image.new('L', (nX, nY), 0)
    ImageDraw.Draw(Im0).polygon(poly, outline=1, fill=1)
    mask = np.array(Im0).astype(bool)
    return(mask)


def mask_to_contour(mask, keep_only_longest_contour = True):
    FoundContours = skm.measure.find_contours(mask, 0.5)
    if keep_only_longest_contour:
        if len(FoundContours) == 1:
            FoundContours = FoundContours[0]
        else:
            L = [len(c) for c in FoundContours]
            im = np.argmax(L)
            FoundContours = FoundContours[im]
            
    return(FoundContours)




# %% 1. Tracking ROIs Subfunctions

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
    
    Mask_inner = contour_to_mask((nY, nX), Contour_inner)
    
    
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

    # Contour_inner = Contour_cell
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


# %% 2. TrackMate Subfunctions

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


def export_as_TrackMate_tracks(tracks,
                               filepath,
                               spaceUnits="pixels",
                               frameInterval=1.0,
                               timeUnits="frames",
                               generationDateTime=None,
                               from_version="TrackMate v7.14.0"):
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
    root = ET.Element("Tracks",
        {
            "nTracks": str(len(tracks)),
            "spaceUnits": str(spaceUnits),
            "frameInterval": str(frameInterval),
            "timeUnits": str(timeUnits),
            "generationDateTime": str(generationDateTime),
            "from": str(from_version),
        })

    # Add each particle and its detections
    for track in tracks:
        track = np.asarray(track)

        if track.size == 0:
            particle = ET.SubElement(root, "particle", {"nSpots": "0"})
            continue

        if track.ndim != 2 or track.shape[1] != 3:
            raise ValueError("Each track must be a 2D array with columns [t, x, y].")

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
                })

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
    
    

def pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                              Dict_TrackMate_Settings = None,
                              ROI_MODE = 'Square ROIs',
                              ROI = None, Mask_cell = None, mask_buffer_um = 0.0, 
                              correct_on_edges_buffer_um = 0,
                              PLOT = False, SHOWPLOT = False, SAVEPLOT = False):
    
    srcDir, tifName = os.path.split(tifPath)
    rawTrackPath = os.path.join(dstDir, rawTrackName)
    
    shape, dtype = tiff_inspect(tifPath)
    nT = shape[0]
    
    if ROI is not None:
        y0, x0 = ROI[0]
        y1, x1 = ROI[1]
        xroi = min(x0, x1)
        yroi = min(y0, y1)
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        x_slice=slice(int(xroi), int(xroi + width))
        y_slice=slice(int(yroi), int(yroi + height))
        
    else:
        x_slice=None
        y_slice=None    
    
    image = load_stack_region(tifPath, time_indices=None, 
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
    print(GREEN + 'Settings: ' + NORMAL, Dict_TrackMate_Settings)
    
    
    # Run TrackMate
    imp = ij.py.to_imageplus(image)
    del(image)
    run_TrackMate(imp, rawTrackPath, PixPerUm, ROI = ROI,
                  **Dict_TrackMate_Settings)
    
    
    # Open the tracks and filters those that are "sticking" to the edges of the ROI
    Tracks = import_TrackMate_tracks(rawTrackPath)
    
    if correct_on_edges_buffer_um > 0:
        buffer_pix = correct_on_edges_buffer_um * PixPerUm
        
        if ROI_MODE == 'Square ROIs':
            Valid_Tracks = []
            for track in Tracks:
                xm = np.median(track[:, 1])
                ym = np.median(track[:, 2])
                dist_x = min(np.abs(xm-0), np.abs(xm-width))
                dist_y = min(np.abs(ym-0), np.abs(ym-height))
                test_x = (dist_x < buffer_pix)
                test_y = (dist_y < buffer_pix)
                if not (test_x or test_y):
                    Valid_Tracks.append(track)
            Tracks = Valid_Tracks
            
        elif ROI_MODE == 'Whole cell':
            Valid_Tracks = []
            Contour_cell = mask_to_contour(Mask_cell)
            Poly_cell = Polygon(shell=Contour_cell)
            Poly_inner_cell = Poly_cell.buffer(-buffer_pix)
            for track in Tracks:
                xm = np.median(track[:, 1])
                ym = np.median(track[:, 2])
                point = MultiPoint([(ym, xm)])
                is_track_in_polygon = Poly_inner_cell.contains(point)
                if is_track_in_polygon:
                    Valid_Tracks.append(track)
            Tracks = Valid_Tracks
        
    # If rectangular ROIs are used, translate tracks coord to be in the referential of the whole image
    if ROI is not None:
        for track in Tracks:
            track[:, 1] += xroi
            track[:, 2] += yroi
            
    # Re-save the cleaned tracks
    export_as_TrackMate_tracks(Tracks, rawTrackPath)
            
    # Plot
    if PLOT:
        if not SHOWPLOT:
            plt.ioff()
        else:
            plt.ion()
            
        setGraphicOptions(mode = 'screen')
        image_raw_0 = load_stack_region(tifPath, time_indices=[0])[0]
        
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
        CL = cL_Set21
        for k in range(len(Tracks)):
            track = Tracks[k]
            color = CL[k%len(CL)]
            ax.plot(track[:,1], track[:,2], ls='-', color=color, lw=0.25)
        
        if SHOWPLOT:
            plt.show()
        
        if SAVEPLOT:
            _, rawTrackName = os.path.split(rawTrackPath)
            figName = rawTrackName.split('.')[0] + '_Plot.png'
            figPath = os.path.join(dstDir, figName)
            fig.savefig(figPath, dpi=500, )
    
    if not SHOWPLOT:
        plt.ion()
    
    
    # Return tracks    
    return(Tracks)
    

  