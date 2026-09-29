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

import sys
import tifffile

import numpy as np


# %% 1. Helper functions

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



# %% 2. Dynamic Differential Microscopy

# %%% DDM Analysis

class ImageStack(object):
    """
    A stack of images on disk with a name pattern like 'mydir/myfile_t{:03d}.tif'
    """
    
    def __init__(self, path):
        """The numbering can start at 0 or 1"""
        self.path = path
        self.t0 = 0
        # get the images shape while checking that the last image do exist
        self.shape, self.type = tiff_inspect(path)
        self.Nbimages = self.shape[0]
        
        # #for some monochrome image format, imread makes 4 channels out of one
        # self.enforceMono = len(self.shape)>2
        # if self.enforceMono:
        #     self.shape = self.shape[:-1]
            
    def __len__(self):
        return(self.Nbimages)
            
    def __getitem__(self, t):
        """
        returns the image at time t
        """
        if t<0: 
            t = len(self)+t
        assert t-self.t0 < self.Nbimages
        
        # im = imread(self.pattern.format(t + self.t0))
        im = load_stack_region(self.path, time_indices=[t])[0]
        # if self.enforceMono:
        #     im = im[...,0]
        return(im)

#### DDM step 1 - spectrumDiff()
# Define the function at the heart of DDM:
# $$\left|\widehat{\Delta I}\right|^2(\vec{q}, t, \Delta t) = \left|\mathcal{F}\left[I(\vec{r}, t+\Delta t) - I(\vec{r}, t)\right]\right|^2$$
# where $I(\vec{r}, t)$ is the intensity of the image at time $t$ at position $\vec{r}$ and $\mathcal{F}$ is the Fourier transform.

def spectrumDiff(im0, im1):
    """
    Compute the squared modulus of the 2D Fourier Transform of 
    the difference between im0 and im1
    """
    # diff = im1-im0.astype(float)
    # FT = np.abs(np.fft.fft2(diff))
    # FT = skm.util.img_as_ubyte(FT)
    # return(FT**2)
    return(np.abs(np.fft.fft2(im1-im0.astype(float)))**2)
    



#### DDM step 2 - timeAveraged()

# A single couple of images is not enough to get good statistics. 
# For a fixed time interval `dt`, we take at most `maxNCouples` 
# couples of images evenly spead in the available range of times.


def timeAveraged(stack, dt, maxNCouples=50):
    """
    Does at most maxNCouples spectreDiff 
    on regularly spaced couples of images. 
    Separation within couple is dt.
    """
    
    #Spread initial times over the available range
    increment = max([(len(stack)-dt)/maxNCouples, 1])
    # print(int(increment))
    initialTimes = np.arange(0, len(stack)-dt, increment, dtype=int)
    
    #perform the time average
    avgFFT = np.zeros(stack.shape[1:])
    for t in initialTimes:
        avgFFT += spectrumDiff(stack[t], stack[t+dt])
        
    return(avgFFT / len(initialTimes))


#### DDM step 3 - RadialAverager()

# Define a class able to perform radial averaging of FFT spectra. 
# For the sake of performance, a RadialAverager instance has a fixed shape 
# and can only process spectra of this shape. This is not a limitation since 
# all the images in a stack do have the same shape.

# Also, since some spectra have anomalously bright cross, we do not take this line 
# and this column into account.

class RadialAverager(object):
    """Radial average of a 2D array centred on (0,0), like the result of fft2d."""
    def __init__(self, shape):
        """A RadialAverager instance can process only arrays of a given shape, fixed at instanciation."""
        assert len(shape)==2
        #matrix of distances
        self.dists = np.sqrt(np.fft.fftfreq(shape[0])[:,None]**2 +  np.fft.fftfreq(shape[1])[None,:]**2)
        #dump the cross
        self.dists[0] = 0
        self.dists[:, 0] = 0
        #discretize distances into bins
        self.bins = np.arange(max(shape)/2 + 1)/float(max(shape))
        #number of pixels at each distance
        self.hd = np.histogram(self.dists, self.bins)[0]
    
    def __call__(self, im):
        """Perform and return(the radial average of the specrum 'im'"""
        assert im.shape == self.dists.shape
        hw = np.histogram(self.dists, self.bins, weights=im)[0]
        return(hw/self.hd)

#### DDM step 4 - logSpaced()

# We won't perform all those steps for every time interval, it would be 
# too time consuming. So we sample time intervals logarithmically.

def logSpaced(L, pointsPerDecade=15):
    """Generate an array of log spaced integers smaller than L"""
    nbdecades = np.log10(L)
    # print(nbdecades, nbdecades * pointsPerDecade)
    return(np.unique(np.logspace(
        start=0, stop=nbdecades, 
        num=int(nbdecades * pointsPerDecade), 
        base=10, endpoint=False
        ).astype(int)))

#### DDM step 5 - ddm()

# Finally, we put everything together to obtain 
# $$\mathcal{D}(\Delta t,q) = \left\langle \left|\widehat{\Delta I}\right|^2 (\vec{q}, t, \Delta t)\right\rangle$$ 
# were $\langle.\rangle$ is the average on initial time $t$ and the orientation of $\vec{q}$.

# Since this can be a long operation, we add a counter

def ddm(stack, idts, maxNCouples=100):
    """Perform time averaged and radial averaged DDM for given time intervals.
    Returns DDM"""
    ra = RadialAverager(stack.shape[1:])
    DDM = np.zeros((len(idts), len(ra.hd)))
    N = len(idts)
    progress_step = N/100
    for i, idt in enumerate(idts):
        DDM[i] = ra(timeAveraged(stack, idt, maxNCouples))
        if i//progress_step > (i-1)//progress_step:
            j = int(i//progress_step)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*(j//5), j))
            sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*20, 100))
    return(DDM)

# %%% Post-treatment

#### DDM step 6 - merge different freqs

def mergeDDM(DDMs, dts, frequencies):
    # Then we merge the two sets of data by scaling the data at 4 Hz so that both values 
    # at 0.25 s are equal. Finally, we average the values of the curves at 4 Hz and 400 Hz 
    # in the first third of their overlap interval.
    
    # Find the closest time at 400Hz to the smallest time at 4Hz
    boundary = np.argmin(np.abs(dts[0] - dts[1][0]))
    
    # find the first third of their overlap
    overlap0 = (len(DDMs[0])-1 - boundary)//3
    overlap1 = np.argmin(np.abs(dts[1] - dts[0][boundary+overlap0]))
    
    # Rescale the value of radial average at 4 Hz according to the value at t=boundary for 400Hz
    overlap_full_1 = (len(DDMs[0])-1 - boundary)//2
    overlap_full_2 = np.argmin(np.abs(dts[1] - dts[0][boundary+overlap_full_1]))
    DDMs[1] *= DDMs[0][boundary+overlap_full_1] / DDMs[1][overlap_full_2]
    
    # interpolate on this first third the DDM at 4Hz on the times at 400Hz
    interpolated = np.transpose([
        np.interp(
            dts[0][boundary:boundary+overlap0],
            dts[1][:overlap1], 
            v)
        for v in DDMs[1][:overlap1].T])
    
    #do a smooth transition on this first third
    x = ((dts[0][boundary:boundary+overlap0]-dts[0][boundary])/(dts[0][boundary+overlap0]-dts[0][boundary]))[:,None]
    transition = (1-x) * DDMs[0][boundary:boundary+overlap0] + x * interpolated
    # Merge 400Hz, transition and 4Hz
    dtMerge = np.concatenate([dts[0][:boundary+overlap0], dts[1][overlap1:]])
    DDMMerge = np.concatenate([DDMs[0][:boundary], transition, DDMs[1][overlap1:]], axis=0)
    return(DDMMerge, dtMerge)
    
