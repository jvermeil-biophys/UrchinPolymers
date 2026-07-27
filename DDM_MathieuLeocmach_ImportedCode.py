# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:14:19 2026

@author: Utilisateur
"""

# %% Image analysis

# Cerbino, R. & Trappe, V. Differential dynamic microscopy: Probing wave vector dependent dynamics with a microscope. Phys. Rev. Lett. 100, 1–4 (2008).


# %% Import necessary libraries

import numpy as np
from scipy.interpolate import splrep, splev
from matplotlib.pylab import imread, imshow, subplot
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt 
# from IPython.display import display

import Libs.UtilityFunctions as ufun


# %% Test import of an image

# DDM is based on the analysis of a video, i.e. a stack of N images stored on the disk. 
# Here we suppose that the filenames follow a pattern like `'mydir/myfile_t{:03d}.tif'`
# so we can obtain the file name at time `t` by doing:

pattern = 'mydir/myfile_t{:03d}.tif'
print(pattern.format(15))

# %% Define ImageStack class

# To manage image reading, keeping track of the number of images, etc. 
# we define a class that we can initialize with the filename pattern 
# and the number of images in the stack. 
# The image at time t can be obtained simply by `stack[t]`. 
# Under the hood, we use `imread` from `matplotlib` that open many format 
# of 8 bits images but fails on 16 bits tiffs.


class ImageStack(object):
    """
    A stack of images on disk with a name pattern like 'mydir/myfile_t{:03d}.tif'
    """
    
    def __init__(self, path):
        """The numbering can start at 0 or 1"""
        self.path = path
        
        # get the images shape while checking that the last image do exist
        self.shape, self.type = ufun.tiff_inspect(path)
        self.Nbimages = self.shape[0]
        
        # #for some monochrome image format, imread makes 4 channels out of one
        # self.enforceMono = len(self.shape)>2
        # if self.enforceMono:
        #     self.shape = self.shape[:-1]
            
    def __len__(self):
        return(self.Nbimages)
            
    def __getitem__(self, t):
        """returns the image at time t"""
        if t<0: t= len(self)+t
        assert t-self.t0 < self.Nbimages
        im = imread(self.pattern.format(t + self.t0))
        if self.enforceMono:
            im = im[...,0]
        return(im)



# class ImageStack(object):
#     """A stack of images on disk with a name pattern like 'mydir/myfile_t{:03d}.tif'"""
#     def __init__(self, pattern, Nbimages, t0=1):
#         """The numbering can start at 0 or 1"""
#         self.pattern = pattern
#         self.t0 = t0
#         self.Nbimages = Nbimages
        
#         #get the images shape while checking that the last image do exist
#         self.shape = imread(pattern.format(Nbimages-1+t0)).shape
        
#         #for some monochrome image format, imread makes 4 channels out of one
#         self.enforceMono = len(self.shape)>2
#         if self.enforceMono:
#             self.shape = self.shape[:-1]
            
#     def __len__(self):
#         return(self.Nbimages)
            
#     def __getitem__(self, t):
#         """returns the image at time t"""
#         if t<0: t= len(self)+t
#         assert t-self.t0 < self.Nbimages
#         im = imread(self.pattern.format(t + self.t0))
#         if self.enforceMono:
#             im = im[...,0]
#         return(im)
        

# %% Test the class and display images

# We try to create such an ImageStack and display some images

srcDir = "C:/Users/Utilisateur/Desktop/DDM-v1.1/TestFiles/SeparateTif/"
fileName = "26-03-02_M1_Pos1_Pa0_C1_Film5min_Dt1sec_crop" 
# "26-03-02_M1_Pos1_Pa0_C1_Film5min_Dt1sec_crop" 
# "26-03-02_M1_Pos1_Pa0_C4_Film5min_Dt1sec_crop"
# "26-03-02_M1_Pos2_Pa0_C2_Film5min_Dt1sec_crop"

stack = ImageStack(u'' + srcDir + fileName + '/' + fileName + '_t{:03d}.tif', 300, t0=0)
plt.figure(figsize=(16,6))
subplot(1,3,1).imshow(stack[0], 'gray')
subplot(1,3,2).imshow(stack[-1], 'gray')
subplot(1,3,3).imshow(stack[-1]-stack[0].astype(float), 'gray')

# %% DDM step 1

# Define the function at the heart of DDM:
# $$\left|\widehat{\Delta I}\right|^2(\vec{q}, t, \Delta t) = \left|\mathcal{F}\left[I(\vec{r}, t+\Delta t) - I(\vec{r}, t)\right]\right|^2$$
# where $I(\vec{r}, t)$ is the intensity of the image at time $t$ at position $\vec{r}$ and $\mathcal{F}$ is the Fourier transform.

def spectrumDiff(im0, im1):
    """
    Compute the squared modulus of the 2D Fourier Transform of 
    the difference between im0 and im1
    """
    return(np.abs(np.fft.fft2(im1-im0.astype(float)))**2)


# Show resulting spectra for $\Delta t=$ 0.01 s, 1 s, 100s.

plt.figure(figsize=(16,6))
I_0_4 = np.fft.fftshift(spectrumDiff(stack[0], stack[4]))
I_0_200 = np.fft.fftshift(spectrumDiff(stack[0], stack[200]))
print(f"{np.percentile(I_0_4, 99):.2e}")
print(f"{np.percentile(I_0_200, 99):.2e}")
vmax=7.1e12
subplot(1,3,1).imshow(I_0_4, 'hot', vmin=0, vmax=vmax)
subplot(1,3,2).imshow(I_0_200, 'hot', vmin=0, vmax=vmax)
#stack2 = ImageStack(u'D:/David/Acquisition/29_05/1103_Souche1/1103_50µL_2h_512x512_4000Im_4/1103_0_{:05d}.tif', 4000)
#subplot(1,3,3).imshow(np.fft.fftshift(spectrumDiff(stack2[0], stack2[400])), 'hot',vmin=0, vmax=2.5e7)

# %% DDM step 2

# A single couple of images is not enough to get good statistics. 
# For a fixed time interval `dt`, we take at most `maxNCouples` 
# couples of images evenly spead in the available range of times.


def timeAveraged(stack, dt, maxNCouples=300):
    """Does at most maxNCouples spectreDiff on regularly spaced couples of images. 
    Separation within couple is dt."""
    #Spread initial times over the available range
    increment = max([(len(stack)-dt)/maxNCouples, 1])
    initialTimes = np.arange(0, len(stack)-dt, increment)
    #perform the time average
    avgFFT = np.zeros(stack.shape)
    for t in initialTimes:
        avgFFT += spectrumDiff(stack[t], stack[t+dt])
    return(avgFFT / len(initialTimes))


plt.figure(figsize=(16,6))
J_0_4 = np.fft.fftshift(timeAveraged(stack, 4))
J_0_10 = np.fft.fftshift(timeAveraged(stack, 10))
J_0_50 = np.fft.fftshift(timeAveraged(stack, 50))
print(f"{np.percentile(J_0_4, 99):.2e}")
print(f"{np.percentile(J_0_50, 99):.2e}")
vmax=7.1e12
subplot(1,3,1).imshow(J_0_4, 'hot', vmin=0, vmax=vmax)
subplot(1,3,2).imshow(J_0_50, 'hot', vmin=0, vmax=vmax)

#a = timeAveraged(stack2, 400)
#subplot(1,3,3).imshow(np.fft.fftshift(a), 'hot',vmin=0, vmax=2.5e7)

# %% DDM step 3

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
        self.dists[:,0] = 0
        #discretize distances into bins
        self.bins = np.arange(max(shape)/2+1)/float(max(shape))
        #number of pixels at each distance
        self.hd = np.histogram(self.dists, self.bins)[0]
    
    def __call__(self, im):
        """Perform and return(the radial average of the specrum 'im'"""
        assert im.shape == self.dists.shape
        hw = np.histogram(self.dists, self.bins, weights=im)[0]
        return(hw/self.hd)

ra = RadialAverager(stack.shape)
plt.plot(ra(J_0_10), 'bo')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('q (px-1)')
plt.ylabel('DDM')


# %% DDM step 4

# We won't perform all those steps for every time interval, it would be 
# too time consuming. So we sample time intervals logarithmically.

def logSpaced(L, pointsPerDecade=15):
    """Generate an array of log spaced integers smaller than L"""
    nbdecades = np.log10(L)
    return(np.unique(np.logspace(
        start=0, stop=nbdecades, 
        num=nbdecades * pointsPerDecade, 
        base=10, endpoint=False
        ).astype(int)))

# %% DDM step 5

# Finally, we put everything together to obtain 
# $$\mathcal{D}(\Delta t,q) = \left\langle \left|\widehat{\Delta I}\right|^2 (\vec{q}, t, \Delta t)\right\rangle$$ 
# were $\langle.\rangle$ is the average on initial time $t$ and the orientation of $\vec{q}$.

# Since this can be a long operation, we add a counter

def ddm(stack, idts, maxNCouples=1000):
    """Perform time averaged and radial averaged DDM for given time intervals.
    Returns DDM"""
    ra = RadialAverager(stack.shape)
    DDM = np.zeros((len(idts), len(ra.hd)))
    #f = FloatProgress(min=0, max=len(idts))
    #display(f)
    for i, idt in enumerate(idts):
        DDM[i] = ra(timeAveraged(stack, idt, maxNCouples))
        #f.value = i
    return(DDM)


# %% A typical DDM analysis

# To cover a wide range of time scales, we have acquired a first stack of 
# 4000 images at 400 Hz and a second similar stack at 4 Hz. 
# With this procedure, we cover time scales between 2.5 ms and 1000 s, 
# but we have to merge the results.
# First, we perform DDM on both stacks. **It will take something like 10 min**


patterns = [u'D:/David/Acquisition/21_05/Colloides/Coll_1%_512x512_4000Im_400/Coll_0_{:05d}.tif',
            u'D:/David/Acquisition/21_05/Colloides/Coll_1%_512x512_4000Im_4/Coll_0_{:05d}.tif']
frequencies = [400,4]
nbimages = 4000
#pixelSize = 6450/10. #in nanometre
pointsPerDecade = 15
maxNCouples = 300 #10 for fast evaluation, 300 for accurate analysis
idts = logSpaced(nbimages, pointsPerDecade)
dts = [idts/float(freq) for freq in frequencies]

DDMs = [ddm(ImageStack(pattern, nbimages), idts, maxNCouples) for pattern in patterns]


for f, d in zip(frequencies, DDMs):
    np.save('DDM{:d}_Colloid.npy'.format(f),d)


# Then we merge the two sets of data by scaling the data at 4 Hz so that both values 
# at 0.25 s are equal. Finally, we average the values of the curves at 4 Hz and 400 Hz 
# in the first third of their overlap interval.


#Find the closest time at 400Hz to the smallest time at 4Hz
boundary = np.argmin(np.abs(dts[0] - dts[1][0]))
#Rescale the value of radial average at 4 Hz according to the value at t=boundary for 400Hz
DDMs[1] *= DDMs[0][boundary] / DDMs[1][0]
#find the first third of their overlap
overlap0 = (len(DDMs[0])-1 - boundary)
overlap1 = np.argmin(np.abs(dts[1] - dts[0][boundary+overlap0]))
#interpolate on this first third the DDM at 4Hz on the times at 400Hz
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


# Test the merging for `q=100`

plt.plot(dts[0], DDMs[0][:,100],'o', label='400 Hz')
plt.plot(dts[1], DDMs[1][:,100],'s', label='4 Hz')
plt.plot(dtMerge, DDMMerge[:,100], label='Merged')
plt.plot(dtMerge, DDMMerge[:,100]*2, 'o', label='Shifted merged')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathcal{D}$')
plt.xlabel(r'$\Delta t\,(s)$')
plt.axvline(dts[0][boundary+overlap0], color='b')
plt.axvline(dts[1][overlap1], color='g')
plt.legend(loc='lower right')

# Now we can export the merged results (adapt to the folder you want to save to)

np.save('DDM_Colloid.npy', DDMMerge)
np.save('dt_Colloid.npy', dtMerge)






