# Purpose: Read in water body data and visualize in python
# Author: Tony Held, tony.held@gmail.com
# Created: 2020-05-14
# Notes:
#    1) Lake volume versus elevation data may not exist to the zero depth (completely dry) point.
#       Water features below the lowest elevation measured will not be considered.
#       For the Loch Lomond analysis, this detail is likely trivial since this only ignores
#       a volume of 1 acre-feet compared to the max volume of 8,646 acre-feet
#       for other water bodies, you may need to extrapolate the lowest measured values to the lake bottom.

# Import numerical array data types and plotting libraries
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

class LakeMeasurements:
    """Class to simulate a lake to characterize its geometry and its water quality properties"""

    # Constants
    SQ_FT_PER_ACRE = (43560,)   # number of square feet in an acre

# Read in lake surface elvation versus lake volume data
# Elevation (ft), Volume (ac-ft)
# 577.51, 8,646 ...
lake_data = np.genfromtxt('../elevation_volume_v2.csv', delimiter=',', skip_header=1, dtype=float)

# Save lake data to renamed variables, convert units, and calculate other properties
# Note that the top layer thickness is less than the other thickness that appear regularly spaced.
elevation = lake_data[:, 0]                 # elevation in feet
vol_acre_feet = lake_data[:, 1]             # total volume in acre-feet
vol_cf = vol_acre_feet * sq_ft_per_acre     # total volume in cubic-feet
vol_inc = np.diff(vol_cf)                   # incremental volume in each layer
thickness = np.diff(elevation)              # thickness of layer

# Find the radius of each layer assuming it is a cylinder
# V = pi*r^2*h
# r = (V/(pi*h))^1/2
radii = (vol_inc / (np.pi * thickness)) ** 0.5

# Determine the coordinates of the center cross section of each disc (aka, cylinder)
# The cross section of a disc will appear as a rectangle with four corners
# The coordinates of each corner is determined as follows

#    (x1, y1)  -------|--------- (y4, x4)
#        |            |              |
#    (x2, y2)  -------|--------- (y3, x3)
#                     CL
# Where:
#    CL - is the center-line of the disc (x=0)
#    (y2-y1) = (y4-y3) = layer thickness
#    x1 = x2= - radius
#    x3 = x4= + radius

y1 = y4 = elevation[:-1]   # Top of disc (excludes bottomost elevation)
y2 = y3 = elevation[1:]    # Bottom of each disc (excludes topmost elevation)
x1 = x2 = - radii
x3 = x4 = + radii

# Combine individual coordinates into a matrix to speed up polygon creation
# Each coordinate component (e.g., x1) is a vector represented as a row
# use vstack to combine data into a 2-d matrix where each row
# contains the 8 data points needed to describe the polygon in the order
# (x1, y1), (x2, y2), (x3, y3), (x4, y4) for each disc

coordinates = np.vstack((x1, y1, x2, y2, x3, y3, x4, y4)).transpose()
(coord_rows, coord_cols) = np.shape(coordinates)
# export coordinates to cvs file for inspection
# np.savetxt("coords.csv", coordinates, delimiter=",")


# Reshape the coordinates so that is is a 3d matrix which stores
# the coordinates in [z, 4, 2] values suitable for plotting functions
verts = np.reshape(coordinates, (-1, 4, 2))
(vert_groups, vert_rows, vert_cols) = np.shape(verts)

# print(f'coordinate rows={coord_rows} col={coord_cols}')
# print(f'first row of coordinates is: {coordinates[0, :]}')

# print(f'verts groups={vert_groups} rows={vert_rows} col={vert_cols}')
# print(f'first groups of  is: {verts[0,:,:]}')

# Create a list of polygons from coordinate data
patches = []  # list to store polygons for plotting lake
for vert in verts:
    polygon = Polygon(vert, closed=True)
    patches.append(polygon)

# store the polygons as a patch collection because this will render faster and
# make it easier to apply a colormap to the polygons
collection = PatchCollection(patches)

# Plot the lake polygons
fig, ax = plt.subplots(1)

ax.add_collection(collection)

# set colors of polygons
cmap = plt.get_cmap('GnBu')
colors = cmap(y1/y1[0])  # scale concentrations on the colormap - just a placeholder for actual data associated with each disc
collection.set_color(colors)
collection.set_edgecolor('k')

ax.autoscale_view()
plt.show()
