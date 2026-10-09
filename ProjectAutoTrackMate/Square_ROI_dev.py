# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:41:09 2026

@author: josep
"""

# %% Import

import time

import numpy as np
import skimage as skm
import matplotlib as mpl
import matplotlib.pyplot as plt

import Libs.UtilityFunctions as ufun

# %% Script

Xc = 300
Yc = 300
Rc = 250
shape = [600, 600]
Angles = np.linspace(0, 2*np.pi, 360)
Xcontour = Xc + Rc*np.cos(Angles)
Ycontour = Yc + Rc*np.sin(Angles)
contour = np.array([Ycontour, Xcontour]).T
mask = ufun.contour_to_mask(shape, contour)

# %%

#### My version
def generate_rois_jv(mask, L, N, seed=None):
    mask_up = np.roll(mask, L, axis=0)
    mask_left = np.roll(mask, L, axis=1)
    mask_upleft = np.roll(mask_up, L, axis=1)
    mask_all = mask & mask_up & mask_left & mask_upleft
    contour_all = np.array(skm.measure.find_contours(mask_all, 0.5))[0]
    
    coords = np.argwhere(mask_all)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, coords.shape[0], 50, dtype=int)
    print(coords[idx,:])
    Ys, Xs = coords[idx,:].T
    
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes[0]
    ax.set_aspect('equal', adjustable='box')
    ax.imshow(mask, cmap='gray')
    ax.plot(contour_all[:,1], contour_all[:,0], 'r-', lw=1)
    for xs, ys in zip(Xs, Ys):
        rect = mpl.patches.Rectangle((xs-L, ys-L), L, L,
                                     facecolor='None', edgecolor='g')
        ax.add_patch(rect)

    ax = axes[1]
    # ax.set_aspect('equal', adjustable='box')
    # ax.imshow(mask, cmap='gray')

    plt.show()
    
    return(coords)

L = 60
N = 3
coords = generate_rois_jv(mask, L, N, seed=None)


#### Chat GPT version

def generate_rois_ai(mask, L, N, seed=None):
    """
    Generate up to N non-overlapping L x L square ROIs
    fully contained within a binary cell mask.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W)
        Binary mask. Nonzero pixels belong to the cell.
    L : int
        Side length of each square, in pixels.
    N : int
        Number of ROIs requested.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    rois : list of tuples
        Each ROI is (x, y, L), where (x, y) is the
        top-left pixel coordinate.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if not isinstance(L, (int, np.integer)) or L <= 0:
        raise ValueError("L must be a positive integer")
    if not isinstance(N, (int, np.integer)) or N < 0:
        raise ValueError("N must be a non-negative integer")

    H, W = mask.shape

    if L > H or L > W or N == 0:
        return []

    rng = np.random.default_rng(seed)

    # 1. Integral image: fast sum over any square
    integral = np.pad(
        mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )

    # 2. Count cell pixels in every possible L x L square
    sums = (
        integral[L:, L:]
        - integral[:-L, L:]
        - integral[L:, :-L]
        + integral[:-L, :-L]
    )

    # 3. Valid positions: every pixel belongs to the cell
    ys, xs = np.where(sums == L * L)

    if len(xs) == 0:
        return []

    # 4. Randomize candidate order
    order = rng.permutation(len(xs))

    # 5. Greedily accept candidates without overlap
    occupied = np.zeros((H, W), dtype=bool)
    rois = []

    for i in order:
        x, y = int(xs[i]), int(ys[i])

        region = occupied[y:y + L, x:x + L]

        if not region.any():
            occupied[y:y + L, x:x + L] = True
            rois.append((x, y, L))

            if len(rois) == N:
                break

    if len(rois) < N:
        print(
            f"Warning: requested {N} ROIs, "
            f"but only {len(rois)} could be placed "
            "in this randomized greedy pass."
        )

    return rois



fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes[0]
ax.set_aspect('equal', adjustable='box')
ax.plot(contour[:,1], contour[:,0], 'k-', lw=1)

ax = axes[1]
ax.set_aspect('equal', adjustable='box')
ax.imshow(mask, cmap='gray')

plt.show()