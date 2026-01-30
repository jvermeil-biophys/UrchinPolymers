# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:21:10 2022
@author: Anumita Jawahar & Joseph Vermeil

UtilityFunctions.py - 
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

source : https://docs.opencv.org/3.4/db/d5b/tutorial_py_mouse_handling.html
"""

# %% Imports

import os
import cv2
import logging
import tifffile
import traceback

import numpy as np
import pandas as pd
import pyjokes as pj
import skimage as skm
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import GraphicStyles as gs
import UtilityFunctions as ufun

# Unused imports

# import shutil
# import pandas as pd

# import warnings
# warnings.filterwarnings("ignore",
#                         category=UserWarning,
#                         module="tifffile")

# %% Functions
     

def getListOfSourceFolders(Dir, forbiddenWords = [], compulsaryWords = []): # 'depthos'
    """
    Given a root folder Dir, search recursively inside for all folders containing .tif images 
    and whose name do not contains any of the forbiddenWords.
    """
    
    res = []
    exclude = False
    for w in forbiddenWords:
        if w.lower() in Dir.lower(): # compare the lower case strings
            exclude = True # If a forbidden word is in the dir name, don't consider it
            
    if exclude or not os.path.isdir(Dir):
        return(res) # Empty list
    
    elif ufun.containsFilesWithExt(Dir, '.tif'):
        # Test the compulsary words only at this final step
        valid = True
        for w in compulsaryWords:
            if w.lower() not in Dir.lower(): # compare the lower case strings
                valid = False # If a compulsary word is NOT in the dir name, don't consider it
            
        if valid:
            res = [Dir] # List with 1 element - the name of this dir
        else:
            return(res)
        
    else:
        listDirs = os.listdir(Dir)
        print(listDirs)
        for D in listDirs:
            path = os.path.join(Dir, D)
            res += getListOfSourceFolders(path, 
                                          forbiddenWords=forbiddenWords,
                                          compulsaryWords=compulsaryWords) 
    # Recursive call to the function !
    # In the end this function will have explored all the sub directories of Dir,
    # searching for folders containing tif files, without forbidden words in their names.        
    return(res)


def copy_metadata_files(ListDirSrc, DirDst, suffix = '.txt'):
    """
    Import the Field.txt files from the relevant folders.
    Calls the copyFilesWithString from ufun with suffix = '_Field.txt'
    """
    for DirSrc in ListDirSrc:
        ufun.copyFilesWithString(DirSrc, DirDst, suffix)
        
def extract_T_from_OMEtiff(ListPaths, DirDst):
    """

    """
    #### TODO !
    pass
        
        
        
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
    
    
def load_IC_region(listpath, time_indices=None, x_slice=None, y_slice=None):
    """
    Load a cropped region of an image collection (list of 2D .tif images) 
    with minimal memory usage.

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

    with tifffile.TiffFile(listpath[0]) as tif:
        series = tif.series[0]   # the first image series
        pages = series.pages
        firstFrame = pages[0]
    if time_indices is None:
        time_indices = np.arange(0, len(listpath))
    if x_slice is None:
        x_slice = slice(0, firstFrame.shape[1])
    if y_slice is None:
        y_slice = slice(0, firstFrame.shape[0])
    cropped_stack = []
    for fp in np.array(listpath)[np.array(time_indices)]:
        with tifffile.TiffFile(fp) as tif:
            series = tif.series[0]   # the first image series
            pages = series.pages
            page = pages[0]
            arr = page.asarray()[y_slice, x_slice]  # crop directly
            cropped_stack.append(arr)
    return(np.stack(cropped_stack, axis=0))


def analyze_cropped_stack(image, tasks = ['Magnet_frames', 'Magnet_pos']):
    """
    
    
    """

    pass

def get_largest_object_contour(img, mode = 'dark'):
    th = skm.filters.threshold_otsu(img)
    if mode == 'dark':
        img_bin = (img < th)
    elif mode == 'bright':
        img_bin = (img > th)
    img_label, num_features = ndi.label(img_bin)
    df = pd.DataFrame(skm.measure.regionprops_table(img_label, img, properties = ['label', 'area']))
    df = df.sort_values(by='area', ascending=False)
    i_label = df.label.values[0]
    img_bin_object = (img_label == i_label)
    img_bin_object = ndi.binary_fill_holes(img_bin_object)
    [contour_object] = skm.measure.find_contours(img_bin_object, 0.5)
    return(contour_object)

def get_magnet_loc(img):
    [nY, nX] = img.shape
    
    #### First thresholding
    contour_magnet = get_largest_object_contour(img, mode = 'dark')
    cmX, cmY = contour_magnet[:,1], contour_magnet[:,0]
    
    #### Second thresholding
    Xmax = np.max(cmX)
    img_cropped = img[:, 0:int(Xmax*1.2)]
    contour_magnet = get_largest_object_contour(img_cropped, mode = 'dark')
    cmX, cmY = contour_magnet[:,1], contour_magnet[:,0]
    
    #### Define circle arc and fit
    selected_points = (np.abs(cmY-(nY/2)) < (nY/4))
    arc_magnet = contour_magnet[selected_points, :]
    mag_center, mag_R = ufun.fitCircle(arc_magnet, loss = 'huber')
    # mag_center in YX format
    return(mag_center, mag_R) #, arc_magnet)


def get_magnet_frames(img):
    [nT, nY, nX] = img.shape
    
    S = np.mean(img, axis=(1,2))
    th = skm.filters.threshold_otsu(S) # unconventionnal use of otsu thresholding
    # low = np.median(S[S < th])
    # high = np.median(S[S > th])
    # th2 = 0.8 * low + 0.2 * high # kind of weighted mean
    # print(th, low, high, th2)
    frames_with_magnet = (S > th)
    if frames_with_magnet[0]: # if the array has True for background instead of False, invert it.
        frames_with_magnet = ~frames_with_magnet

    first_idx = ufun.findFirst(1, frames_with_magnet)
    last_idx = ufun.findLast(1, frames_with_magnet)
    
    first_frame_withMagnet = first_idx + 1
    last_frame_beforeMagnet = first_frame_withMagnet - 1
    last_frame_withMagnet = last_idx + 1
    
    # Maybe unecessary
    # if frames_with_magnet[-1]:
    #     last_frame_withMagnet = nT
    
    return(last_frame_beforeMagnet, last_frame_withMagnet)
    



def Z_projection(stack, kind = 'min', scaleFactor = 1/4, output_type = 'uint8', normalize = False):
    """
    From an image stack in the (T, Y, X) format
    does a scaled-down (by 'scaleFactor') Z-projection (minimum by default)
    to display the best image for cropping boundary selection.
    """
    dtype_matching = {'uint8':cv2.CV_8U,
                      'uint16':cv2.CV_16U}
    
    imgWidth, imgHeight = stack.shape[2], stack.shape[1]
    if kind == 'min':
        Zimg = np.min(stack, axis = 0)
    elif kind == 'max':
        Zimg = np.max(stack, axis = 0)
    elif kind == 'median':
        Zimg = np.median(stack, axis = 0)
    Zimg = cv2.resize(Zimg, (int(imgWidth*scaleFactor), int(imgHeight*scaleFactor)))
    if normalize:
        Zimg = cv2.normalize(Zimg, None, 0, 255, cv2.NORM_MINMAX, dtype=dtype_matching[output_type])
        # Zimg = cv2.normalize(Zimg, None, 0, 65535, cv2.NORM_MINMAX, dtype=dtype_matching['uint16'])
    return(Zimg)


# def shape_selection_V0(event, x, y, flags, param):
#     """
#     Non-interactive rectangular selection.
#     Has to be called in cv2.setMouseCallback(StackPath, shape_selection)
#     """
    
#     # grab references to the global variables
#     global ref_point, crop, allZimg #, iZ

#     # if the left mouse button was clicked, record the starting
#     # (x, y) coordinates and indicate that cropping is being performed
#     if event == cv2.EVENT_LBUTTONDOWN:
#         ref_point = [[x, y]]

#     # check to see if the left mouse button was released
#     elif event == cv2.EVENT_LBUTTONUP:
#         # record the ending (x, y) coordinates and indicate that
#         # the cropping operation is finished
#         ref_point.append([x, y])

#         # draw a rectangle around the region of interest
#         cv2.rectangle(allZimg[i], ref_point[0], ref_point[1], (0, 255, 0), 1)

  
def shape_selection(event, x, y, flags, param):
    """
    Interactive rectangular selection.
    Has to be called in cv2.setMouseCallback(StackPath, shape_selection)
    """
    
    # Grab references to the global variables
    global ix, iy, drawing, ref_point, crop, img, img_copy
    
    # If the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        img_copy = np.copy(img)
        ref_point = [[x, y]]
    
    # If the mouse moves, reinitialize the image and the rectangle to match 
    # the current position
    elif event == cv2.EVENT_MOUSEMOVE: 
        
        if drawing == True:
            img = np.copy(img_copy)
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append([x, y])
        # Final rectangle around the region of interest
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
        


def crop_and_copy(DirSrc, DirDst, allRefPoints, allStackPaths, 
                source_format = 'single file', suffix = '',
                bin_output = False, bin_N = 1, bin_func = np.mean,
                channel = 'nan', prefix = 'nan'):
    """
    Using user specified rectangular coordinates from the previous functions,
    Crop cells stack and copy them onto the destination file.
    
    If you are using Metamorph for imaging, you will have to update the 'prefix' and 'channel'
    variables depending on the names of your images. 
    It follows the usual, default metamorph naming system: 'Prefix_Channel_Timepoint0.tif'
    If you are using labview, the default will be 'nan' for both variables.
    """
    
    count = 0
    N_suffix = 0
    suffix_1 = ''
    suffix_2 = suffix
    
    for i in range(len(allStackPaths)):
    # for refPts, stackPath in zip(allRefPoints, allStackPaths):
        
        stackPath = allStackPaths[i]
        stackDir, stackName = os.path.split(stackPath)
        
        refPts = np.array(allRefPoints[i])
        x1, x2 = int(min(refPts[:,0])), int(max(refPts[:,0]))
        y1, y2 = int(min(refPts[:,1])), int(max(refPts[:,1]))
        
        # to detect supplementary selections
        try:
            if (allStackPaths[i-1]==allStackPaths[i]):
                N_suffix = N_suffix + 1
                suffix_1 = '-' + str(N_suffix)
            else:
                N_suffix = 0
                suffix_1 = ''
        except:
            N_suffix = 0
            suffix_1 = ''
            
        print(gs.BLUE + 'Loading '+ stackPath +'...' + gs.NORMAL)
        
        try:
            if source_format == 'single file':
                FilesList = os.listdir(stackPath)
                TifList = [f for f in FilesList if f.endswith('.tif')]
                if len(TifList) != 1:
                    print(gs.BRIGHTRED + '/! Several images in the folder in single file mode' + gs.NORMAL)
                    continue
                else:
                    stackPath = os.path.join(stackPath, TifList[0])
                    stackShape, stackType = tiff_inspect(stackPath)
                    (nT, nY, nX) = stackShape

                    # To avoid that the cropped region gets bigger than the image itself
                    x1, x2, y1, y2 = max(0, x1), min(nX, x2), max(0, y1), min(nY, y2)
                    cropped_stack = load_stack_region(stackPath, time_indices=None, 
                                              x_slice=slice(x1,x2,1), y_slice=slice(y1,y2,1))
                    
            elif source_format == 'image collection':
                FilesList = os.listdir(stackPath)
                TifList = [f for f in FilesList if f.endswith('.tif')]
                if len(TifList) <= 1:
                    print(gs.BRIGHTRED + '/! Single image in the folder in image collection mode' + gs.NORMAL)
                    continue
                else:
                    stackPaths = [os.path.join(stackPath, f) for f in TifList]
                    stackShape, stackType = tiff_inspect(stackPaths[0])
                    nT = len(stackPaths)
                    (nY, nX) = stackShape
                    # To avoid that the cropped region gets bigger than the image itself
                    x1, x2, y1, y2 = max(0, x1), min(nX, x2), max(0, y1), min(nY, y2)
                    cropped_stack = load_IC_region(stackPaths, time_indices=None, 
                                              x_slice=slice(x1,x2,1), y_slice=slice(y1,y2,1))
            
            if bin_output:
                cropped_stack = skm.measure.block_reduce(cropped_stack, 
                                                     block_size = bin_N, func = bin_func, 
                                                     cval = 0)

            FileDst = stackName + suffix_1 + suffix_2 + '.tif'
            skm.io.imsave(os.path.join(DirDst, FileDst), cropped_stack)
            print(gs.GREEN + os.path.join(DirDst, FileDst) + '\nSaved sucessfully' + gs.NORMAL)
        
        except Exception:
            traceback.print_exc()
            print(gs.RED + os.path.join(DirDst, FileDst) + '\nError when saving' + gs.NORMAL)
            continue
        
        if count%5 == 0:
            joke = pj.get_joke(language='en', category= 'all')
            print(joke)
            
        count = count + 1
        


#%% Define parameters


DirSrc = 'C:/Users/Utilisateur/Desktop/MicroscopeData/Leica/25-09-19/M2' #'/M4_patterns_ctrl' // \\M1_depthos
DirDst = 'C:/Users/Utilisateur/Desktop/MicroscopeData/Leica/25-09-19/Crops'
DirDst_bins = 'C:/Users/Utilisateur/Desktop/MicroscopeData/Leica/25-09-19/Binned'

microscope = 'Leica'
source_format = 'single file' # 'image collection'
# imagePrefix = 'im'
checkIfAlreadyExist = True
GetOMEdata = True

scaleFactor = 1/8

forbiddenWords = ['capture', 'captures', 'crop', 'crops', 'croped']
compulsaryWords = []

# Disable the Warnings from TiffFile
logging.getLogger('tifffile').setLevel(logging.ERROR)

# One of these lines would reactivate it
# logging.getLogger('tifffile').setLevel(logging.INFO)
# logging.getLogger('tifffile').setLevel(logging.WARNING)

#%% Main function 1/2

allStackPaths = getListOfSourceFolders(DirSrc,
                                       forbiddenWords = forbiddenWords,
                                       compulsaryWords = compulsaryWords)

allStacks = []
allStacksToCrop = []
allStacksPath = []
ref_point = []
allRefPoints = []
allZimg = []
allZimg_og = []

checkIfAlreadyExist = True

if not os.path.exists(DirDst):
    os.mkdir(DirDst)

print(gs.BLUE + 'Constructing all Z-Projections...' + gs.NORMAL)

for i in range(len(allStackPaths)):
    print(i)
    StackFolder = allStackPaths[i]
    StackFolderDir, StackFolderName = os.path.split(StackFolder)
    validStackFolder = True
        
    if not ufun.containsFilesWithExt(StackFolder, '.tif'):
        validStackFolder = False
        print(gs.BRIGHTRED + '/! Is not a valid stack' + gs.NORMAL)
        
    elif checkIfAlreadyExist and os.path.isfile(os.path.join(DirDst, StackFolderName + '.tif')):
        validStackFolder = False
        print(gs.GREEN + ':-) Has already been copied' + gs.NORMAL)
        
    if validStackFolder:
        
        if source_format == 'single file':
            FilesList = os.listdir(StackFolder)
            TifList = [f for f in FilesList if f.endswith('.tif')]
            if len(TifList) != 1:
                print(gs.BRIGHTRED + '/! Several images in the folder in single file mode' + gs.NORMAL)
                continue
            else:
                stackPath = os.path.join(StackFolder, TifList[0])
                allStacksPath.append(stackPath)
                stackShape, stackType = tiff_inspect(stackPath)
                (nT, nY, nX) = stackShape
                TT = np.array([t for t in range(0, nT, 5)])
                stack = load_stack_region(stackPath, time_indices=TT, 
                                          x_slice=None, y_slice=None)
                
        elif source_format == 'image collection':
            FilesList = os.listdir(StackFolder)
            TifList = [f for f in FilesList if f.endswith('.tif')]
            if len(TifList) <= 1:
                print(gs.BRIGHTRED + '/! Single image in the folder in image collection mode' + gs.NORMAL)
                continue
            else:
                stackPaths = [os.path.join(StackFolder, f) for f in TifList]
                stackShape, stackType = tiff_inspect(stackPaths[0])
                nT = len(stackPaths)
                (nY, nX) = stackShape
                TT = np.array([t for t in range(0, nT, 5)])
                stack = load_IC_region(stackPaths, time_indices=TT, 
                                          x_slice=None, y_slice=None)
                
        Zimg = Z_projection(stack, kind = 'min', scaleFactor = scaleFactor, normalize = True)
        allStacks.append(StackFolder)
        allZimg.append(Zimg)
        print(gs.CYAN + '--> Will be copied' + gs.NORMAL)
        # except:
        #     print(gs.BRIGHTRED + '/!\ Unexpected error during file handling' + gs.NORMAL)


# copy_metadata_files(allStacks, DirDst, suffix = '_Status.txt')
# copy_metadata_files(allStacks, DirDst, suffix = '_Status.txt')
# allZimg_og = np.copy(np.asarray(allZimg)) # TBC

#%% Main function 2/2

instructionText = "Draw the ROIs to crop !\n\n(1) Click on the image to define a rectangular selection\n"
instructionText += "(2) Press 'a' to accept your selection, 'r' to redraw it, "
instructionText += "or 's' if you have a supplementary selection to make (you can use 's' more than once per stack !)\n"
instructionText += "(3) Make sure to choose the number of files you want to crop at once\nin the variable 'limiter'"
instructionText += "\n\nLet's gooooo !\n"

#Change below the number of stacks you want to crop at once. Run the code again to crop the remaining files. 
# !! WARNING: Sometimes choosing too many can make your computer bug !!
limiter = 15

print(gs.YELLOW + instructionText + gs.NORMAL)

# if reset == 1:
    
#     allZimg = np.copy(allZimg_og)
#     ref_point = []
#     allRefPoints = []

count = 0
# for i in range(len(allZimg)):
for i in range(min(len(allZimg), limiter)):
    
    stackPath = allStacks[i]
    stackDir, stackName = os.path.split(stackPath)
    
    Nimg = len(allZimg)
    ncols = 5
    nrows = 3
    # nrows = ((Nimg-1) // ncols) + 1
    
    if count%(ncols*nrows) == 0:
        count = 0
    
    # test
    ix,iy = 0, 0
    drawing = False
    img = allZimg[i]
    img_backup = np.copy(img)
    img_copy = np.copy(img)
    
    cv2.namedWindow(stackName)
    cv2.moveWindow(stackName, (count//nrows)*340, count%nrows*350)
    
    # cv2.setMouseCallback(StackPath, shape_selection_V0)
    cv2.setMouseCallback(stackName, shape_selection)
    
    while True:
    # Display the image and wait for a keypress
        stackPath = allStacks[i]
        stackDir, stackName = os.path.split(stackPath)
        cv2.imshow(stackName, img)
        key = cv2.waitKey(20) & 0xFF
        
    # Press 'r' to reset the crop
        if key == ord("r"):  
            img = np.copy(img_backup)  
             
    # If the 'a' key is pressed, break from the loop and move on to t/he next file
        elif key == ord("a"):
            allRefPoints.append(np.asarray(ref_point)/scaleFactor)
            allStacksToCrop.append(stackPath)
            break
        
    # If the 's' key is pressed, save the coordinates and rest the crop, ready to save once more
    # The code can accept more than 2 selections per stack !
        elif key == ord("s"):
            allRefPoints.append(np.asarray(ref_point)/scaleFactor)
            allStacksToCrop.append(stackPath)
            img = np.copy(img_backup)     
        
    count = count + 1
    print(stackPath)
    
cv2.destroyAllWindows()

print(allStacksToCrop)
print(gs.BLUE + 'Saving all tiff stacks...' + gs.NORMAL)

crop_and_copy(DirSrc, DirDst, allRefPoints[:], allStacksToCrop[:], 
            source_format = source_format, suffix = '',
            bin_output = False, bin_N = 1, bin_func = np.mean,
            channel = 'nan', prefix = 'nan')

if GetOMEdata:
    extract_T_from_OMEtiff(allStacksPath, DirDst)

# crop_and_copy(DirSrc, DirDst, allRefPoints[:], allStacksToCrop[:], 
#             source_format = source_format, suffix = '_Binned',
#             bin_output = True, bin_N = 3, bin_func = np.mean,
#             channel = 'nan', prefix = 'nan')

# skm.measure.block_reduce(image, block_size=2, func=<function sum>, cval=0, func_kwargs=None)


# %% Tests

# %%%

dirPath = 'C:/Users/Joseph/Desktop/WorkingData/LeicaData/25-12-18_WithJessica/25-12-18_Droplet01_JN-Magnet_MyOne-Gly80'
fileName = '25-12-18_20x_FastBFGFP_Droplet_1_MMStack_Default.ome.tif'
filePath = os.path.join(dirPath, fileName)

result, case = ufun.OMEDataParser(filePath)
print(case)













# %%%

# %%%

srcDir = "C:/Users/Utilisateur/Desktop/AnalysisPulls/26-01-14_BeadTracking/Films"
imgName = '26-01-14_M1_C3_Pa1_P2.tif'
imgPath = os.path.join(srcDir, imgName)


print(tiff_inspect(imgPath))

img_size, img_type = tiff_inspect(imgPath)

# If = load_stack_region(imgPath)
# img = If
# img = skm.util.invert(img)
# fI, fL = get_magnet_frames(img)

# fig, ax = plt.subplots(1, 1)
# ax.imshow(img[fI], cmap='gray')
# plt.show()

# fig, ax = plt.subplots(1, 1)
# ax.imshow(img[fL], cmap='gray')
# plt.show()



Ic = load_stack_region(imgPath, x_slice=slice(0,50,1))
img = Ic
img = skm.util.invert(img)
fI, fL = get_magnet_frames(img)

fig, ax = plt.subplots(2, 2, sharex = True, sharey= True, figsize=(3, 8))
ax[0,0].imshow(img[fI-1], cmap='gray')
ax[0,1].imshow(img[fI], cmap='gray')
ax[1,0].imshow(img[fL-1], cmap='gray')
ax[1,1].imshow(img[fL], cmap='gray')
# fig.tight_layout()
fig.suptitle(imgName[:-4], fontsize=10)
plt.show()

# S = np.mean(img, axis=(1,2))


# %%%

A = np.ones(50)
for i in range(20, 40):
    A[i] = 1.5

th = skm.filters.threshold_otsu(A)
print(th)


A = np.ones((500, 400, 400))
S = np.sum(A, axis=(1,2))

# %%%

srcDir = "C:/Users/Utilisateur/Desktop/AnalysisPulls/26-01-14_BeadTracking/Films"
imgPath = os.path.join(srcDir, '26-01-14_M1_C4_Pa1_P3.tif')


print(tiff_inspect(imgPath))

I1 = load_stack_region(imgPath)
I1 = skm.util.invert(I1)

I1_min = Z_projection(I1, kind='min', scaleFactor=1, output_type='uint16')

center, R, contour = get_magnet_loc(I1_min)



# fig, ax = plt.subplots(1, 1)
# ax.plot(contour[:,1], contour[:,0], 'b--')
# ax.plot(center[0], center[1], 'go', markersize = 10)
# circle = plt.Circle((center[0], center[1]), R, facecolor='None', edgecolor='r')
# ax.add_patch(circle)
# ax.set_aspect('equal')
# plt.show()

fig, ax = plt.subplots(1, 1)
ax.imshow(I1_min, cmap='gray')
ax.plot(contour[:,1], contour[:,0], 'b--')
ax.plot(center[1], center[0], 'go', markersize = 10)
circle = plt.Circle((center[1], center[0]), R, facecolor='None', edgecolor='r')
ax.add_patch(circle)
plt.show()


# %%%

TestFolder = "C:/Users/Utilisateur/Desktop/MicroscopeData/Leica/25-09-19/Test"

stackPath = os.path.join(TestFolder, 'M1_D6_P1_S', '25-09-19_M1_D6_P1_noUV_Gly75p_NaSS5p_I2959-50mM_1_MMStack_Default.ome.tif')
listfiles = os.listdir(os.path.join(TestFolder, 'M1_D6_P1_IC'))
listpaths = [os.path.join(TestFolder, 'M1_D6_P1_IC', f) for f in listfiles]

time_indices = np.arange(10, 50, 1)
x_slice = slice(0, 512+1024, 1)
y_slice = slice(700, 800+400, 1)

print(tiff_inspect(stackPath))

I1 = load_stack_region(stackPath,
                       time_indices=time_indices, x_slice=x_slice, y_slice=y_slice)

I1_median = Z_projection(I1, kind='median', scaleFactor=1, output_type='uint16')

I1_cleaned = I1 - I1_median

plt.imshow(I1_cleaned[-1])

# I2 = load_IC_region(listpaths, 
#                    time_indices=time_indices, x_slice=x_slice, y_slice=y_slice)

# skm.io.imshow(I1[0])
# skm.io.imshow(I2[0])




