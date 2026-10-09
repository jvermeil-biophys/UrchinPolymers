# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:41:09 2026

@author: josep
"""

# %% Import

import time
import shapely

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


Xc = 410
Yc = 410
Rc = 50
shape = [600, 600]
Angles = np.linspace(0, 2*np.pi, 360)
Xcontour = Xc + Rc*np.cos(Angles)
Ycontour = Yc + Rc*np.sin(Angles)
excluded_contour = np.array([Ycontour, Xcontour]).T

# %%

#### My version
def generate_rois_jv_mask(mask, L, N, seed=None):
    mask_up = np.roll(mask, L, axis=0)
    mask_left = np.roll(mask, L, axis=1)
    mask_upleft = np.roll(mask_up, L, axis=1)
    mask_all = mask & mask_up & mask_left & mask_upleft
    contour_all = np.array(skm.measure.find_contours(mask_all, 0.5))[0]
    
    coords = np.argwhere(mask_all)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, coords.shape[0], N, dtype=int)
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

L = 40
N = 10
top = time.time()
coords = generate_rois_jv_mask(mask, L, N, seed=None)
print(f'{time.time()-top:.3f} s')



# %% My version - Shapely

def make_valid_poly(poly, L, it=5):
    Lstep = L/it
    all_polys = [poly]
    for i in range(1, it+1):
        poly_cell_down = shapely.affinity.translate(poly, xoff=-i*Lstep, yoff=0)
        poly_cell_right = shapely.affinity.translate(poly, xoff=0, yoff=-i*Lstep)
        poly_cell_downright = shapely.affinity.translate(poly, xoff=-i*Lstep, yoff=-i*Lstep)
        all_polys += [poly_cell_down, poly_cell_right, poly_cell_downright]

    return shapely.intersection_all(all_polys)

def exclude_region_from_poly(poly, xy, L):
    L *= 1.001
    x, y = xy
    x0, y0 = x, y
    x1, y1 = x+L, y+L
    XY_square = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
    square = shapely.Polygon(XY_square)
    return shapely.difference(poly, square)

def get_random_point_in_polygon(poly, seed=None):
     minx, miny, maxx, maxy = poly.bounds
     rng = np.random.default_rng(seed)
     while True:
         p = shapely.Point(rng.integers(minx, maxx, dtype=int), rng.integers(miny, maxy, dtype=int))
         if poly.contains(p):
             return p

def generate_rois_jv_shape(contour, L, N, excluded_contour = None,
                           seed = None, PLOT = False):
    poly_cell = shapely.Polygon(contour)
    if not (excluded_contour is None):
        poly_cell = shapely.difference(poly_cell, shapely.Polygon(excluded_contour))
        
    poly_valid = make_valid_poly(poly_cell, L)
    
    if PLOT:
        fig, axes = plt.subplots(1, (N+1), figsize=(3*(N+1), 3))
        ax = axes[0]
        # xy_cell = np.array(list(poly_cell.exterior.xy)).T
        # xy_valid = np.array(list(poly_valid.exterior.xy)).T
        ax.set_aspect('equal', adjustable='box')
        shapely.plotting.plot_polygon(poly_cell, ax=ax, color='red', facecolor='None', 
                                      add_points=False, linewidth=1)
        shapely.plotting.plot_polygon(poly_valid, ax=ax, color='blue', facecolor='None', 
                                      add_points=False, linewidth=1)
    
    list_rois = []
    
    for k in range(N):
        p = get_random_point_in_polygon(poly_valid, seed=seed)
        x0, y0 = p.x, p.y
        x1, y1 = x0+L, y0+L
        list_rois.append([[x0, y0], [x1, y1]])
        
        poly_cell = exclude_region_from_poly(poly_cell, (x0, y0), L)
        poly_valid = make_valid_poly(poly_cell, L)
        
        
        if PLOT:
            ax = axes[k+1]
            ax.set_aspect('equal', adjustable='box')
            
            for i in range(len(list_rois)):
                xy0, xy1 = list_rois[i]
                x0, y0 = xy0
                if i == len(list_rois)-1:
                    ax.plot([x0], [y0], 'ro', zorder=12)
                else:
                    ax.plot([x0], [y0], 'go', zorder=11)
                rect = mpl.patches.Rectangle((x0, y0), L, L,
                                             facecolor='None', edgecolor='g',
                                             lw=2, zorder=10)
                # ax.add_patch(rect)
                
                
            shapely.plotting.plot_polygon(poly_cell, ax=ax, color='red', facecolor='None', 
                                          add_points=False, linewidth=1)
            shapely.plotting.plot_polygon(poly_valid, ax=ax, color='blue', facecolor='None', 
                                          add_points=False, linewidth=1)

    if PLOT:
        plt.show()

    ROIs = np.array(list_rois)
    return ROIs

L = 120
N = 3
top = time.time()
coords = generate_rois_jv_shape(contour, L, N, excluded_contour = excluded_contour, 
                                seed=None, PLOT = True) 
print(f'{time.time()-top:.3f} s')


# %% Chat GPT version

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