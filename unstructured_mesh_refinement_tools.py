#!usr/bin/env python3
"""
Function suite for the processing steps
of the unstrucured mesh refinement method
"""
import numpy as np
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import shapes
from scipy import stats
from skimage import morphology, measure
from rdp import rdp # pip install rdp
import pandas as pd

#==================================================
# Steps of the Unstructured Mesh Refinement Method

# Functions are listed in their order of operations
# See workflow figure for details
#==================================================
def generate_boundary(bounds, dx, extent, buffer=5):
    """Function to generate a raster boundary mask from an ANUGA model
    boundary.
    Inputs:
        bounds (np.ndarray) : List of vertices defining the ANUGA model
            boundary, e.g. from np.loadtxt('boundary.csv')
        dx (float) : Grid size of the input imagery raster
        extent (list or array) : geographical extent of imagery raster,
            specified as extent = [xmin, xmax, ymin, ymax]
        buffer (int) : Radius of a disk-shaped buffer to apply between the
            boundary and interior features, specified in number of pixels
    Outputs:
        outside (np.ndarray) : Binary raster same size as the imagery
            raster with 1's representing cells outside the boundary (with
            buffer) and 0's inside the boundary
    """
    # Coordinates of raster
    xvect = np.arange(extent[0], extent[1], dx)
    yvect = np.arange(extent[2], extent[3], dx)
    gridX, gridY = np.meshgrid(xvect, yvect)
    gridXY_array = np.array([np.concatenate(gridX),
                             np.concatenate(gridY)]).transpose()
    gridXY_array = np.ascontiguousarray(gridXY_array)

    # Use matplotlib's Path to find cells inside polygon
    path = matplotlib.path.Path(bounds)
    outside = ~path.contains_points(gridXY_array)

    outside.shape = (len(yvect), len(xvect))
    outside = np.flipud(outside)

    # Border mask with a buffer
    if buffer > 0:
        outside = morphology.dilation(outside, morphology.disk(buffer))
    return outside

def enforce_connectivity(image, outside):
    """
    Function to filter out any objects not connected to the largest
    cluster in image.
    Inputs:
        image (np.ndarray) : Binary input raster with 1's corresponding
            to hydrodynamically active areas, which get mapped to high
            resolution, and 0's otherwise
        outside (np.ndarray) : Binary raster same size as the imagery
            raster with 1's representing cells outside the boundary
            and 0's inside the boundary. Output of generate_boundary()
    Outputs:
        connected_cluster (np.ndarray) : Binary raster similar to
            input image, with only the largest cluster of 1's remaining
    """
    image[outside] = 0 # Mask boundary
    # Count the unique objects in the image:
    image_count = np.array(measure.label(image, connectivity=2), dtype=float)
    image_count[image_count==0] = np.nan
    # Find the most common object index:
    largest_val = stats.mode(image_count, nan_policy='omit')[0][0]
    # Filter out everything outside largest connected cluster:
    connected_cluster = (image_count==largest_val)
    return connected_cluster

def ensure_minimum_width(connected_cluster, buffer=5):
    """
    Function to enforce minimum dimension of objects in active network
    Inputs:
        connected_cluster (np.ndarray) : Binary raster indicating
            largest connected cluster of active pixels, output of
            enforce_connectivity()
        buffer (int) : Radius of a disk-shaped buffer to apply between the
            interior features, specified in number of pixels
    Outputs:
        dilated_cluster (np.ndarray) : Binary raster similar to
            connected_cluster, with buffer applied
    """
    # Dilate objects to min size
    dilated_cluster = morphology.dilation(connected_cluster,
                                          morphology.disk(buffer))
    return dilated_cluster

def smooth_object_interface(dilated_cluster, outside, buffer=5):
    """
    Function to smooth the interface between channels and island
    Inputs:
        dilated_cluster (np.ndarray) : Binary raster indicating
            filtered connected cluster of active pixels, output of
            ensure_minimum_width()
        outside (np.ndarray) : Binary raster same size as the imagery
            raster with 1's representing cells outside the boundary
            and 0's inside the boundary. Output of generate_boundary()
        buffer (int) : Radius of a disk-shaped buffer with which to
            smooth interior features, specified in number of pixels
    Outputs:
        smoothed_cluster (np.ndarray) : Binary raster similar to
            dilated_cluster, with smoothing applied
    """
    # Invert outside before closing
    dilated_cluster[outside] = 1
    # Smooth edges of objects
    smoothed_cluster = morphology.binary_closing(dilated_cluster,
                                                 morphology.disk(buffer))
    return smoothed_cluster

def ensure_large_islands(smoothed_cluster, min_size):
    """
    Function to eliminate islands below a threshold size
    Inputs:
        smoothed_cluster (np.ndarray) : Binary raster indicating
            filtered connected cluster of active pixels, output of
            smooth_object_interface()
        min_size (int) : Minimum area (specified in pixels) of islands
            to qualify for coarsening
    Outputs:
        large_islands (np.ndarray) : Binary raster similar to
            smoothed_cluster, with islands below min_size eliminated
    """
    large_islands = morphology.remove_small_holes(smoothed_cluster,
                                                  min_size, 2)
    return large_islands

def smooth_island_interface(large_islands, outside, buffer=4):
    """
    Optional function to smooth channel-island interface from opposite
    orientation as smooth_object_interface. Can lead to slightly more
    locally-convex perimeter.
    Inputs:
        large_islands (np.ndarray) : Binary raster indicating
            largest islands for coarsening, output of ensure_large_islands()
        outside (np.ndarray) : Binary raster same size as the imagery
            raster with 1's representing cells outside the boundary
            and 0's inside the boundary. Output of generate_boundary()
        buffer (int) : Radius of a disk-shaped buffer with which to
            smooth island features, specified in number of pixels
    Outputs:
        large_islands (np.ndarray) : Binary raster similar to
            large_islands, with islands slightly smoothed
    """
    large_islands = morphology.binary_opening(large_islands,
                                              morphology.disk(buffer))
    # Make sure nothing encroached on the boundary in the process:
    large_islands[outside] = 1
    return large_islands

def raster2polygon(large_islands, img_path):
    """
    Function to convert binary raster with 0's indicating islands
    into a list of vector polygons.
    Inputs:
        large_islands (np.ndarray) : Binary raster indicating
            largest islands for coarsening, output of ensure_large_islands()
            or smooth_island_interface()
        img_path (str) : Path to original input image, from which we need to
            grab the geo transform matrix
    Outputs:
        polycoords (list) : List of polygon vertices outlining the islands
            (holes) of large_islands
    """
    # Highlight holes as objects now:
    image = np.array(large_islands==0).astype(np.uint8)
    src = rasterio.open(img_path)

    # Find polygon vertices using rasterio
    with rasterio.Env():
        results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(
            rasterio.features.shapes(image, mask=None,
                                     transform=src.transform)))

    geoms = list(results)
    polycoords = [geoms[c]['geometry']['coordinates'][0] for c in range(len(geoms)) if geoms[c]['properties']['raster_val'] == 1]
    return polycoords

def simplify_polygons(polycoords, epsilon):
    """
    Function to decimate the vertices of a list of polygons using the
    Ramer-Douglas-Peucker algorithm.
    Inputs:
        polycoords (list) : List of polygon vertices outlining the islands
            to be simplified, output of raster2polygon()
        epsilon (float or int) : Epsilon value to use for the RDP algorithm,
            essentially a buffer lengthscale to eliminate proximal vertices
    Outputs:
        simple_polygons (list) : Simplified (decimated) form of polycoords
    """
    # Simplify with Ramer-Douglas-Peucker algorithm
    simple_polygons = [rdp(c, epsilon=epsilon) for c in polycoords]
    return simple_polygons

def getAngle(a, b, c):
    """
    Helper function for filter_poly_angles()
    Find angle between three points ABC
    """
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) \
                       - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def removeAcute(polygon):
    """
    Helper function for filter_poly_angles()
    Remove any angles which are too sharp (< 28 deg)
    """
    newpoly = polygon.copy()
    # Work backwards to keep indexing correct
    for n in range(len(polygon)-2, 0, -1):
        ang = getAngle(polygon[n-1], polygon[n], polygon[n+1])
        if (ang < 28) | (ang > 332):
            del newpoly[n]
    return newpoly

def filter_poly_angles(simple_polygons):
    """
    Function to eliminate acute angles from of a list of polygons 
    Inputs:
        simple_polygons (list) : List of polygon vertices outlining islands
            to be simplified, output of simplify_polygons()
    Outputs:
        safe_simple_polygons (list) : List similar to simple_polygons with
            sharp angles removed
    """
    # Remove any unsafe angles too sharp for Anuga
    safe_simple_polygons = [removeAcute(c) for c in simple_polygons]
    return safe_simple_polygons

def save_for_anuga(safe_simple_polygons, outfolder, triangle_res):
    """
    Function to save list of polygons to CSV files indicating their future
    ANUGA resolution in outfolder.
    Inputs:
        safe_simple_polygons (list) : List of filtered polygon vertices
            outlining islands to be coarsened
        outfolder (str) : String specifying folder path in which to
            save polygon files
        triangle_res (float or list of floats) : Max triangle area
            to be assigned to this polygon when loaded into ANUGA, which
            is saved into the filename for ease of use later. Can be
            specified as a single float for all polygons or as a list/array
            of floats of equal length to the number of polygons.
    Outputs:
        Saves a list of CSV files in outfolder
    """
    for n, simple_poly in enumerate(safe_simple_polygons):
        try:
            res = triangle_res[n]
        except TypeError:
            res = triangle_res
        name = os.path.join(outfolder,'CoarseReg%s_Res%s.csv' % (n, res))
        df = pd.DataFrame(data=simple_poly[0:-1]) # Delete redundant last data point
        df.to_csv(name, index=False, header=False)
    
    # NOTE: Code snippet to load these back in for use with ANUGA later:
    # polygon_files = glob.glob('outfolder/*Reg*.csv')
    # inside_regions = []
    # for poly in polygon_files:
    #    polyres = int(poly.split('_Res')[-1].replace('.csv',''))
    #    inside_regions.append([anuga.read_polygon(poly), polyres])
    # domain = anuga.create_domain_from_regions(..., interior_regions=inside_regions)
    return

def plot_polygons(polygons, fill=True, outline=False, outline_color='k'):
    """
    Helper function to plot the vector form of the interior polygons,
    either filled in or as outlines.
    Inputs:
        polygons (list) : List of polygon coordinates
        fill (bool) : Option to plot polygons as filled-in
        outline (bool) : Option to plot polygon outlines
        outline_color (str) : If outline is True, plot with this color
    Outputs:
        Outputs a figure showing polygon features
    """
    fig = plt.figure(figsize=(8, 8), dpi=400)
    for poly in polygons:
        x = [c[0] for c in poly]
        y = [c[1] for c in poly]
        if fill:
            plt.fill(x,y)
        if outline:
            plt.plot(x,y,c=outline_color, linewidth=0.5, alpha=0.9)
    plt.axis('scaled')
    return
