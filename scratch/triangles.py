# Scratch Code

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np

fig,ax = plt.subplots(1)

N = 10
nfloors = np.random.rand(N) # some random data

patches = []

cmap = plt.get_cmap('RdYlBu')
colors = cmap(nfloors) # convert nfloors to colors that we can use later
print(f'nfloors=\n{nfloors}')
print(f'colors=\n{colors}')

for i in range(N):
    verts = np.random.rand(3,2)+i # random triangles, plus i to offset them
    polygon = Polygon(verts,closed=True)
    patches.append(polygon)

collection = PatchCollection(patches)

ax.add_collection(collection)

collection.set_color(colors)

ax.autoscale_view()
plt.show()