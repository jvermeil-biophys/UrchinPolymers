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
import pyjokes as pj
import matplotlib.pyplot as plt

import GraphicStyles as gs
import UtilityFunctions as ufun

from skimage import io


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


def copyMetadataFiles(ListDirSrc, DirDst, suffix = '.txt'):
    """
    Import the Field.txt files from the relevant folders.
    Calls the copyFilesWithString from ufun with suffix = '_Field.txt'
    
    """
    for DirSrc in ListDirSrc:
        ufun.copyFilesWithString(DirSrc, DirDst, suffix)
        
        
        
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


def Zprojection(stack, kind = 'min', scaleFactor = 1/4, normalize = False):
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
        Zimg = cv2.normalize(Zimg, None, 0, 255, cv2.NORM_MINMAX, dtype=dtype_matching['uint8'])
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
        


def cropAndCopy(DirSrc, DirDst, allRefPoints, allStackPaths, 
                mode = 'single file', channel = 'nan', prefix = 'nan'):
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
    suffix = ''
    
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
                suffix = '-' + str(N_suffix)
            else:
                N_suffix = 0
                suffix = ''
        except:
            N_suffix = 0
            suffix = ''
            
        print(gs.BLUE + 'Loading '+ stackPath +'...' + gs.NORMAL)
        
        try:
            if mode == 'single file':
                FilesList = os.listdir(StackFolder)
                TifList = [f for f in FilesList if f.endswith('.tif')]
                if len(TifList) != 1:
                    print(gs.BRIGHTRED + '/! Several images in the folder in single file mode' + gs.NORMAL)
                    continue
                else:
                    stackPath = os.path.join(StackFolder, TifList[0])
                    stackShape, stackType = tiff_inspect(stackPath)
                    (nT, nY, nX) = stackShape

                    # To avoid that the cropped region gets bigger than the image itself
                    x1, x2, y1, y2 = max(0, x1), min(nX, x2), max(0, y1), min(nY, y2)
                    cropped_stack = load_stack_region(stackPath, time_indices=None, 
                                              x_slice=slice(x1,x2,1), y_slice=slice(y1,y2,1))
                    
            elif mode == 'image collection':
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
                    # To avoid that the cropped region gets bigger than the image itself
                    x1, x2, y1, y2 = max(0, x1), min(nX, x2), max(0, y1), min(nY, y2)
                    cropped_stack = load_IC_region(stackPaths, time_indices=None, 
                                              x_slice=slice(x1,x2,1), y_slice=slice(y1,y2,1))
            
            FileDst = stackName + suffix + '.tif'
            io.imsave(os.path.join(DirDst, FileDst), cropped_stack)
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
DirDst = 'C:/Users/Utilisateur/Desktop/MicroscopeData/Leica/25-09-19/M2/Crops'

microscope = 'Leica'
mode = 'single file' # 'image collection'
# imagePrefix = 'im'
checkIfAlreadyExist = True

scaleFactor = 1/8

forbiddenWords = ['capture', 'captures', 'crop', 'crops']
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
    print(StackFolderName)
        
    if not ufun.containsFilesWithExt(StackFolder, '.tif'):
        validStackFolder = False
        print(gs.BRIGHTRED + '/! Is not a valid stack' + gs.NORMAL)
        
    elif checkIfAlreadyExist and os.path.isfile(os.path.join(DirDst, StackFolderName + '.tif')):
        validStackFolder = False
        print(gs.GREEN + ':-) Has already been copied' + gs.NORMAL)
        
    if validStackFolder:
        
        if mode == 'single file':
            FilesList = os.listdir(StackFolder)
            TifList = [f for f in FilesList if f.endswith('.tif')]
            if len(TifList) != 1:
                print(gs.BRIGHTRED + '/! Several images in the folder in single file mode' + gs.NORMAL)
                continue
            else:
                stackPath = os.path.join(StackFolder, TifList[0])
                stackShape, stackType = tiff_inspect(stackPath)
                (nT, nY, nX) = stackShape
                TT = np.array([t for t in range(0, nT, 5)])
                stack = load_stack_region(stackPath, time_indices=TT, 
                                          x_slice=None, y_slice=None)
                
        elif mode == 'image collection':
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
                
        Zimg = Zprojection(stack, kind = 'min', scaleFactor = scaleFactor, normalize = True)
        allStacks.append(StackFolder)
        allZimg.append(Zimg)
        print(gs.CYAN + '--> Will be copied' + gs.NORMAL)
        # except:
        #     print(gs.BRIGHTRED + '/!\ Unexpected error during file handling' + gs.NORMAL)


# copyMetadataFiles(allStacks, DirDst, suffix = '_Status.txt')
# copyMetadataFiles(allStacks, DirDst, suffix = '_Status.txt')
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
    # display the image and wait for a keypress
        stackPath = allStacks[i]
        stackDir, stackName = os.path.split(stackPath)
        cv2.imshow(stackName, img)
        key = cv2.waitKey(20) & 0xFF
        
    # press 'r' to reset the crop
        if key == ord("r"):  
            img = np.copy(img_backup)  
             
    # if the 'a' key is pressed, break from the loop and move on to t/he next file
        elif key == ord("a"):
            allRefPoints.append(np.asarray(ref_point)/scaleFactor)
            allStacksToCrop.append(stackPath)
            break
        
    # if the 's' key is pressed, save the coordinates and rest the crop, ready to save once more
    # The code can accept more than 2 selections per stack !!!
        elif key == ord("s"):
            allRefPoints.append(np.asarray(ref_point)/scaleFactor)
            allStacksToCrop.append(stackPath)
            img = np.copy(img_backup)     
        
    count = count + 1
    
cv2.destroyAllWindows()

print(gs.BLUE + 'Saving all tiff stacks...' + gs.NORMAL)

cropAndCopy(DirSrc, DirDst, allRefPoints[:], allStacksToCrop[:], mode = mode)



# %% Tests

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

I1_median = Zprojection(I1, kind='median', scaleFactor=1, output_type='uint16')

I1_cleaned = I1 - I1_median

plt.imshow(I1_cleaned[-1])

# I2 = load_IC_region(listpaths, 
#                    time_indices=time_indices, x_slice=x_slice, y_slice=y_slice)

# io.imshow(I1[0])
# io.imshow(I2[0])




