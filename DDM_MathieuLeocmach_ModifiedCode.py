# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:14:19 2026

@author: Utilisateur
"""

# %% Image analysis

# Cerbino, R. & Trappe, V. Differential dynamic microscopy: Probing wave vector dependent dynamics with a microscope. Phys. Rev. Lett. 100, 1–4 (2008).


# %% Import necessary libraries

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 

from scipy.interpolate import splrep, splev
from matplotlib.pylab import imread, imshow, subplot
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
# from IPython.display import display

import Libs.UtilityFunctions as ufun



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
        self.t0 = 0
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
        """
        returns the image at time t
        """
        
        if t<0: 
            t = len(self)+t
            
        assert t-self.t0 < self.Nbimages
        
        # im = imread(self.pattern.format(t + self.t0))
        im = ufun.load_stack_region(self.path, time_indices=[t])[0]
        
        # if self.enforceMono:
        #     im = im[...,0]
        
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

# srcDir = "C:/Users/Utilisateur/Desktop/DDM-v1.1/TestFiles/SeparateTif/"
# fileName = "26-03-02_M1_Pos1_Pa0_C1_Film5min_Dt1sec_crop" 
# "26-03-02_M1_Pos1_Pa0_C1_Film5min_Dt1sec_crop" 
# "26-03-02_M1_Pos1_Pa0_C4_Film5min_Dt1sec_crop"
# "26-03-02_M1_Pos2_Pa0_C2_Film5min_Dt1sec_crop"
# stack = ImageStack(u'' + srcDir + fileName + '/' + fileName + '_t{:03d}.tif', 300, t0=0)

# srcDir = "F://IntraCellTracking//26-06-19_FastAcq"
srcDir = "F://IntraCellTracking//26-07-29_FastAcq_Fecondation//Crops"
fileName = "26-07-29_PostF_2min_Pos11_10fps_Texp100ms1_CSU642_crop.tif"
filePath = os.path.join(srcDir, fileName)

stack = ImageStack(filePath)

plt.figure(figsize=(16,6))
subplot(1,3,1).imshow(stack[0], 'gray')
subplot(1,3,2).imshow(stack[-1], 'gray')
subplot(1,3,3).imshow(stack[-1]-stack[0].astype(float), 'gray')

# %% DDM step 1 - spectrumDiff()

# Define the function at the heart of DDM:
# $$\left|\widehat{\Delta I}\right|^2(\vec{q}, t, \Delta t) = \left|\mathcal{F}\left[I(\vec{r}, t+\Delta t) - I(\vec{r}, t)\right]\right|^2$$
# where $I(\vec{r}, t)$ is the intensity of the image at time $t$ at position $\vec{r}$ and $\mathcal{F}$ is the Fourier transform.

def spectrumDiff(im0, im1):
    """
    Compute the squared modulus of the 2D Fourier Transform of 
    the difference between im0 and im1
    """
    return(np.abs(np.fft.fft2(im1-im0.astype(float)))**2)

# %% Test spectrumDiff()
# Show resulting spectra for $\Delta t=$ X s, Y s, Z s.

# plt.figure(figsize=(15,5))
# I_0_40 = np.fft.fftshift(spectrumDiff(stack[0], stack[40-1]))
# I_0_400 = np.fft.fftshift(spectrumDiff(stack[0], stack[400-1]))
# I_0_4000 = np.fft.fftshift(spectrumDiff(stack[0], stack[4000-1]))
# print(f"{np.percentile(I_0_40, 99):.2e}")
# print(f"{np.percentile(I_0_400, 99):.2e}")
# print(f"{np.percentile(I_0_4000, 99):.2e}")
# V1, V2, V3 = np.percentile(I_0_40, 99), np.percentile(I_0_400, 99), np.percentile(I_0_4000, 99)
# vmax=3.1e11
# subplot(1,3,1).imshow(I_0_40, 'hot', vmin=0, vmax=V1)
# subplot(1,3,2).imshow(I_0_400, 'hot', vmin=0, vmax=V2)
# subplot(1,3,3).imshow(I_0_4000, 'hot', vmin=0, vmax=V3)


plt.figure(figsize=(15,5))
I_0_20 = np.fft.fftshift(spectrumDiff(stack[0], stack[20-1]))
I_0_200 = np.fft.fftshift(spectrumDiff(stack[0], stack[200-1]))
I_0_2000 = np.fft.fftshift(spectrumDiff(stack[0], stack[2000-1]))
print(f"{np.percentile(I_0_20, 99):.2e}")
print(f"{np.percentile(I_0_200, 99):.2e}")
print(f"{np.percentile(I_0_2000, 99):.2e}")
V1, V2, V3 = np.percentile(I_0_20, 99), np.percentile(I_0_200, 99), np.percentile(I_0_2000, 99)
vmax=3.1e11
subplot(1,3,1).imshow(I_0_20, 'hot', vmin=0, vmax=V1)
subplot(1,3,2).imshow(I_0_200, 'hot', vmin=0, vmax=V2)
subplot(1,3,3).imshow(I_0_2000, 'hot', vmin=0, vmax=V3)

# %% DDM step 2 - timeAveraged()

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
        # print(t+dt)
        avgFFT += spectrumDiff(stack[t], stack[t+dt])
    return(avgFFT / len(initialTimes))

# %% Test timeAveraged() 

J_0_5 = timeAveraged(stack, 5)
J_0_10 = timeAveraged(stack, 10)
J_0_50 = timeAveraged(stack, 50)

# %% Plot timeAveraged() 

V1, V2, V3 = np.percentile(J_0_5, 99), np.percentile(J_0_10, 99), np.percentile(J_0_50, 99)

plt.figure(figsize=(16, 6))

subplot(1,3,1).imshow(np.fft.fftshift(J_0_5), 'hot', vmin=0, vmax=V1)
subplot(1,3,2).imshow(np.fft.fftshift(J_0_10), 'hot', vmin=0, vmax=V2)
subplot(1,3,3).imshow(np.fft.fftshift(J_0_50), 'hot', vmin=0, vmax=V3)

plt.show()


#a = timeAveraged(stack2, 400)
#subplot(1,3,3).imshow(np.fft.fftshift(a), 'hot',vmin=0, vmax=2.5e7)

# %% DDM step 3 - RadialAverager()

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

# %% Test the RadialAverager

ra = RadialAverager(stack.shape[1:])
plt.figure(figsize=(4, 4))
plt.plot(ra(J_0_5), 'b-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('q (px-1)')
plt.ylabel('DDM')
plt.show()

# %% DDM step 4 - logSpaced()

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

# %% DDM step 5 - ddm()

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


# %% A typical DDM analysis

# %%% 1. Compute

# To cover a wide range of time scales, we have acquired a first stack of 
# 4000 images at 400 Hz and a second similar stack at 4 Hz. 
# With this procedure, we cover time scales between 2.5 ms and 1000 s, 
# but we have to merge the results.
# First, we perform DDM on both stacks. **It will take something like 10 min**

srcDir = "F://IntraCellTracking//26-06-19_FastAcq"
fileNames = ["FilmBF_fastAcq_4000f_200Hz_C3.tif", "FilmBF_fastAcq_4000f_10Hz_C3_ROI.tif"]
srcDir = "F://IntraCellTracking//26-07-29_FastAcq_Fecondation//Crops"
fileNames = ["26-07-29_PostF_2min_Pos11_10fps_Texp100ms1_CSU642_crop.tif"]

# filePath = os.path.join(srcDir, fileName)

paths = [os.path.join(srcDir, fN) for fN in fileNames]
frequencies = [10] #[200, 10]
nbimages = 2000 # 4000
#pixelSize = 6450/10. #in nanometre
pointsPerDecade = 15
maxNCouples = 30 #10 for fast evaluation, 300 for accurate analysis
idts = logSpaced(nbimages, pointsPerDecade)
dts = [idts/float(freq) for freq in frequencies]

# patterns = [u'D:/David/Acquisition/21_05/Colloides/Coll_1%_512x512_4000Im_400/Coll_0_{:05d}.tif',
#             u'D:/David/Acquisition/21_05/Colloides/Coll_1%_512x512_4000Im_4/Coll_0_{:05d}.tif']
# DDMs = [ddm(ImageStack(pattern, nbimages), idts, maxNCouples) for pattern in patterns]
# for f, d in zip(frequencies, DDMs):
#     np.save('DDM{:d}_Colloid.npy'.format(f),d)

DDMs = []
for p in paths:
    print(f'\n\nAnalyzing {os.path.split(p)}...')
    DDM = ddm(ImageStack(p), idts, maxNCouples)
    DDMs.append(DDM)
    
for f, d in zip(frequencies, DDMs):
    np.save('DDM{:d}_fastAcq_PostF-2min_Ncouples10.npy'.format(f),d)


# %%% 2. Merge

srcDir = "F://IntraCellTracking//26-06-19_FastAcq//DDM_C3_300Nc"
fileNames = ["DDM200_fastAcq_C3.npy", "DDM10_fastAcq_C3.npy"]

frequencies = [200, 10]
DDMs = [np.load(os.path.join(srcDir, fN)) for fN in fileNames]

# Then we merge the two sets of data by scaling the data at 4 Hz so that both values 
# at 0.25 s are equal. Finally, we average the values of the curves at 4 Hz and 400 Hz 
# in the first third of their overlap interval.

# Find the closest time at 400Hz to the smallest time at 4Hz
boundary = np.argmin(np.abs(dts[0] - dts[1][0]))
# Rescale the value of radial average at 4 Hz according to the value at t=boundary for 400Hz
DDMs[1] *= DDMs[0][boundary] / DDMs[1][0]
# find the first third of their overlap
overlap0 = (len(DDMs[0])-1 - boundary)
overlap1 = np.argmin(np.abs(dts[1] - dts[0][boundary+overlap0]))
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

# %%% 3. Plot

DDM_plot = DDMs[0]
dt_plot = dts[0]

# Test the merging for `q=100`
SCALE_40X_Leica = 4.3725
PixPerUm_40X_Leica = 1/SCALE_40X_Leica

SCALE_60X_W1 = 9.26
PixPerUm_60X_W1 = 1/SCALE_60X_W1

N_pix = 512
L_um = N_pix*PixPerUm_60X_W1
dq = 2*np.pi / L_um
qmax = 11.7
QQ = np.arange(1, 1+len(DDM_plot[0,:]))*dq
Q_plot = [10, 30, 100]

# for q in Q_plot:
#     fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#     ax.plot(dts[0], DDMs[0][:,q],'o', label='200 Hz')
#     ax.plot(dts[1], DDMs[1][:,q],'s', label='10 Hz')
#     ax.plot(dtMerge, DDM_plot[:,q], label='Merged')
#     # ax.plot(dtMerge, DDM_plot[:,100]*2, 'o', label='Shifted merged')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_ylabel(r'$\mathcal{D}$')
#     ax.set_xlabel(r'$\Delta t\,(s)$')
#     ax.axvline(dts[0][boundary], color='r')
#     ax.axvline(dts[0][boundary+overlap0], color='b')
#     ax.axvline(dts[1][overlap1], color='g')
#     ax.legend(loc='lower right')
#     ax.set_title(f'q = {q:.0f}')
#     plt.show()

(Ndt, Nq) = DDM_plot.shape
fig, axes = plt.subplots(2, 1, figsize = (5, 8))
ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$q\ (\mu m^{-1})$')
ax.set_ylabel('$D$')
for i in range(0, Ndt, 10):
    ax.plot(QQ, DDM_plot[i,:], marker='.', ls='',
            color = mpl.cm.autumn(i/Ndt))
    ax.axvline(dq, color='gray', ls='-', alpha=0.7)
    ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
    
fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=np.min(dt_plot), vmax=np.max(dt_plot)), 
                                   cmap="autumn"),
             ax=ax, label="$\Delta t$")
    

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\Delta t\ (s)$')
ax.set_ylabel('$D$')
for j in range(0, Nq, 25):
    ax.plot(dt_plot, DDM_plot[:,j], marker='.', ls='',
            color = mpl.cm.winter(j/Nq))
    
fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=np.min(QQ), vmax=np.max(QQ)), 
                                   cmap="winter"),
             ax=ax, label="$q$")

plt.show()

# Now we can export the merged results (adapt to the folder you want to save to)

# np.save('DDM_fastAcq_C3.npy', DDM_plot)
# np.save('dt_fastAcq_C3.npy', dtMerge)

ApB_est = np.median(DDM_plot[-4:,:], axis=0)
B_est = np.median(DDM_plot[:6,:], axis=0)
B_best = 4.1e10

fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
for ax in axes:
    ax.set_xscale('log')
    ax.set_yscale('log')
ax = axes[0]
ax.plot(QQ, ApB_est, 'r.')
ax.axvline(dq, color='gray', ls='-', alpha=0.7)
ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
ax = axes[1]
ax.plot(QQ, B_est,'k.')
ax.axvline(dq, color='gray', ls='-', alpha=0.7)
ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
ax = axes[2]
ax.plot(QQ, ApB_est-B_est,'b.')
ax.plot(QQ, ApB_est-B_best,'g.')
ax.axvline(dq, color='gray', ls='-', alpha=0.7)
ax.axvline(qmax, color='gray', ls='-', alpha=0.7)

plt.show()

# %%% 4. Use a model to fit A, B and get f (Brownian case)

DDM_fit = DDMs[0]
dt_fit = dts[0]

# Test the merging for `q=100`
SCALE_40X_Leica = 4.3725
PixPerUm_40X_Leica = 1/SCALE_40X_Leica

SCALE_60X_W1 = 9.26
PixPerUm_60X_W1 = 1/SCALE_60X_W1

N_pix = 512
L_um = N_pix*PixPerUm_60X_W1
dq = 2*np.pi / L_um
qmin = 1.5
qmax = 10 # 11.7
QQ = np.arange(1, 1+len(DDM_plot[0,:]))*dq


def simple_brownian_model(dt, A, B, G):
    D = A * (1 - np.exp(-G*dt)) + B
    return(D)

valid_iQ, valid_Q = [], []
list_A, list_B, list_G = [], [], []

for iq in range(len(QQ)):
    q = QQ[iq]
    if q >= qmin and q < qmax:
        valid_Q.append(q)
        valid_iQ.append(iq)
        
        D = DDM[:,iq]
        dt = dt_fit
        
        # some initial parameter values - must be within bounds
        initB = np.median(DDM_fit[:3,iq], axis=0)/10
        initA = np.median(DDM_fit[-4:,iq], axis=0) - initB
        initG = 1
        
        print(f'{initB:.2e}')
        
        initialParameters = [initA, initB, initG]
        
        # bounds on parameters - initial parameters must be within these
        lowerBounds = (0, 0, 0)
        upperBounds = (np.inf, np.inf, np.inf)
        parameterBounds = [lowerBounds, upperBounds]
        
        params, covM = curve_fit(simple_brownian_model, dt, D, 
                                 p0=initialParameters, bounds = parameterBounds)
        
        print(f'>> {params[1]:.2e}')
        if params[1] < 1e8:
            params[1] = initB
        
        list_A.append(params[0])
        list_B.append(params[1])
        list_G.append(params[2])
        
        
X, Y = np.log(valid_Q), np.log(list_G)
params, results = ufun.fitLineHuber(X, Y)
    

# %%% Plot the fit

DDM_fit = DDMs[0]
dt_fit = dts[0]

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax = ax
ax.set_xscale('log')
ax.set_yscale('log')
cmap = mpl.cm.plasma

idx = slice(0, len(valid_iQ), 10)
k = 0

for iq, A, B, G in zip(valid_iQ[idx], list_A[idx], list_B[idx], list_G[idx]):
    q = QQ[iq]
    D = DDM_fit[:, iq]
    color = cmap(k/len(valid_iQ[idx]))
    k += 1
    ax.plot(dt, D, ls='', marker='o', color = color, label=f'q = {q:.3f}')
    
    D_fit = simple_brownian_model(dt, A, B, G)
    ax.plot(dt, D_fit, ls='-', marker='', color = color, label=f'fit, B = {B:.1e}')

ax.legend()
ax.grid()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    
cmap = mpl.cm.viridis

idx = slice(0, len(valid_iQ), 10)
k = 0

for iq, A, B, G in zip(valid_iQ[idx], list_A[idx], list_B[idx], list_G[idx]):
    q = QQ[iq]
    D = DDM_fit[:, iq]
    color = cmap(k/len(valid_iQ[idx]))
    k += 1
    fR = 1 - ((D-B)/A)
    fR_fit = np.exp(-G*dt)
    
    ax = axes[0]
    ax.plot(dt, fR, ls='', marker='o', color = color, label=f'q = {q:.3f}')
    ax.plot(dt, fR_fit, ls='-', marker='', color = color, label=f'fit, G = {G:.1e}')
    
    ax = axes[1]
    ax.plot(dt*q*q, fR, ls='', marker='o', color = color, label=f'q = {q:.3f}')
    ax.plot(dt*q*q, fR_fit, ls='-', marker='', color = color, label=f'fit, G = {G:.1e}')

for ax in axes:
    ax.set_xscale('log')
    ax.legend()
    ax.grid()
    
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(10,4))

ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(valid_Q, list_G, ls='', marker='o')
ax.grid()

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(valid_Q, list_A, ls='', marker='o')
ax.plot(valid_Q, list_B, ls='', marker='o')

plt.show()


