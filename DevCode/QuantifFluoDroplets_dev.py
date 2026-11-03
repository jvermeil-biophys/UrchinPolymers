# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 16:14:53 2025

@author: Utilisateur
"""


# %% 1. Imports

import os
import skimage as skim
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile

# %% 2. Subfunctions

# Function to segment droplets
def segment_droplets(img):
    val = skim.filters.threshold_otsu(img)
    mask = img > val
    labels = skim.measure.label(mask)
    regions = skim.measure.regionprops(labels, intensity_image=img)
    return(regions)

def get_circularity(region):
    area = region.area_filled
    peri = region.perimeter
    circularity = 4 * np.pi * (area/(peri**2))
    return(circularity)

# Function to filter droplets based on circularity and surface area criteria
def filter_droplets(regions, circ_min = 0.8, area_min = 3*1e4, area_max = 7*1e4):
    filtered_regions = [r for r in regions if circ_min < get_circularity(r) < 1 and area_min < r.area_filled < area_max]
    return(filtered_regions)

# Function to measure gray value statistics
def measure_gray_values(region, img):
    (min_row, min_col, max_row, max_col) = region.bbox
    mask = region.image
    droplet_pixels = img[min_row:max_row, min_col:max_col][mask > 0]
    results = {}
    results['mean_gray_value'] = np.mean(droplet_pixels)
    results['std_gray_value'] = np.std(droplet_pixels)
    results['median_gray_value'] = np.median(droplet_pixels)
    results['min_gray_value'] = np.min(droplet_pixels)
    results['max_gray_value'] = np.max(droplet_pixels)
    return(results)

# %% 3. Main function

# Main function to process images
def process_images(directory, label = '', verbose = False):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            images = tifffile.imread(os.path.join(directory, filename))
            for i, img in enumerate(images):
                regions = segment_droplets(img)
                filtered_regions = filter_droplets(regions)
                for region in filtered_regions:
                    stats = measure_gray_values(region, img)
                    stats['label'] = label
                    stats['page'] = i+1
                    results.append(stats)
                    if verbose == True:
                        print(f'Image: {filename}, Page: {i+1}, Droplet stats:')
                        for k in stats.keys():
                            print(k, f'{stats[k]:.0f}')
                        print('\n')
    return(results)
                    
                    
# %% 4. Script

res = []

# Directory containing .tif images
mainDir = "E:/WorkingData/LeicaData/25-12-05/DextranDrops/"
listDir = [os.path.join(mainDir, p) for p in os.listdir(mainDir) if os.path.isdir(os.path.join(mainDir, p))]
for p in listDir:
    root, name = os.path.split(p)
    res += process_images(p, label = name)
    
listDf = [pd.DataFrame({k:[v] for k,v in d.items()}) for d in res]
df = pd.concat(listDf)

# %% Filter

fltrs = [df['label'] != '1_25']

global_filter = fltrs[0]
try:
    for f in fltrs[1:]:
        global_filter = global_filter & f
except:
    pass

df_f = df[global_filter]

# %% Plot

fig, ax = plt.subplots(1, 1)
sns.swarmplot(data=df_f, ax=ax, 
              x='label', y='median_gray_value')
sns.boxplot(data=df_f, ax=ax, 
              x='label', y='median_gray_value')
plt.show()



