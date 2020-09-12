# scratch Code

import matplotlib.pyplot as plt
# Based on examples from:
# https://matplotlib.org/3.2.1/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

#plt.ion()
plt.ioff()
print(f'interactive mode is: {plt.isinteractive()}')

# data common to all examples
data_length = 10
x1 = np.linspace(0, 20, data_length)
x2 = np.linspace(10, 30, data_length)
y1 = np.random.randint(0, 10, data_length)
y2 = np.random.randint(0, 10, data_length) * 3

if False:
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], linewidth=3, color='red')  # Plot some data on the axes.

if False:
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'g^--')
    plt.axis([-10, 10, -20, 20])
    plt.show()

if False:
    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()

if False:
    data = {'a': np.arange(50),
            'c': np.random.randint(0, 50, 50),
            'd': np.random.randn(50)}
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100

    plt.scatter('a', 'b', c='c', s='d', data=data)
    plt.xlabel('entry a')
    plt.ylabel('entry b')
    plt.show()

if False:
    names = ['group_a', 'group_b', 'group_c']
    values = [1, 10, 100]

    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)
    plt.suptitle('Categorical Plotting')
    plt.show()

if False:
    # A Python program to return multiple
    # values from a method using tuple

    # This function returns a tuple
    def fun1():
        str = "geeksforgeeks"
        x = 20
        return str, x  # Return tuple, we could also write (str, x)

    # This function returns a list
    def fun2():
        str = "geeksforgeeks"
        x = 20
        return [str]  # Return tuple, we could also write (str, x)


    # Driver code to test above method
    str2  = fun1()  # Assign returned tuple
    print(str2)

    str3, = fun2()  # Assign returned list
    print(str3)


if False:
    x = np.linspace(0, 100, 15)
    y = x**2
    # plt.plot(x, y, linewidth=2.0)
    # plt.show()

    line1 = plt.plot(x, y, '-')
    print(type(line1))
    print(len(line1))

    for i in line1:
        print(type(i))

    [line2] = plt.plot(x, y, '-')
    print(type(line2))

    line2.set_antialiased(False)  # turn off antialiasing
    plt.show()

if False:
    lines = plt.plot(x1, y1, 'ro:', x2, y2, 'g^--')
    # use keyword args
    plt.setp(lines, linewidth=2.0)
    # plt.setp(lines)
    plt.show()

if False:
    def f(t):
        return np.exp(-t) * np.cos(2 * np.pi * t)


    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)

    plt.figure(3)
    plt.subplot(211)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

    plt.subplot(212)
    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')

    plt.figure(1.1)
    plt.show()

if False:
    import matplotlib.pyplot as plt

    plt.figure(1)  # the first figure
    plt.subplot(211)  # the first subplot in the first figure
    plt.plot([1, 2, 3])
    plt.subplot(212)  # the second subplot in the first figure
    plt.plot([4, 5, 6])

    plt.figure(2)  # a second figure
    plt.plot([4, 5, 6])  # creates a subplot(111) by default

    plt.figure(1)  # figure 1 current; subplot(212) still current
    plt.subplot(211)  # make subplot(211) in figure1 current
    plt.title('Easy as 1, 2, 3')  # subplot 211 title
    plt.show()

if False:
    # Data for plotting
    x1 = np.linspace(0.0, 5.0)
    x2 = np.linspace(0.0, 2.0)
    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    y2 = np.cos(2 * np.pi * x2)

    # Create two subplots sharing y axis
    fig, (ax1, ax2) = plt.subplots(2, sharey=True, sharex=True)
#    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(x1, y1, 'ko-')
    ax1.set(title='A tale of 2 subplots', ylabel='Damped oscillation')

    ax2.plot(x2, y2, 'r.-')
    ax2.set(xlabel='time (s)', ylabel='Undamped')

    plt.show()

if False:
    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)

    # the histogram of the data
    plt.figure(1)
    n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)
    plt.title('density=1')
    plt.figure(2)
    n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)
    plt.title('density=True')
    plt.figure(3)
    n, bins, patches = plt.hist(x, 50, facecolor='g', alpha=0.75)
    plt.title('no density')

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()

if False:
    x = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [2, 2, 2],
                  [2, 2, 2]])
    x1= x[:,0]
    y = np.array([[1, 2, 3],
                  [2, 3, 4],
                  [2, 3, 4],
                  [1, 2, 3]])
    c = np.linspace(0,1, 4)
    c1 = c.transpose()

    fig1, ax1 = plt.subplots()
    ax1.fill(x1, y, c)


    plt.figure(2)
    plt.fill(x1, y, c1)
    plt.Polygon
    plt.show()

if False:
    # test out colormapping
    x = np.linspace(0, 1, 20)
    y = x/20
    print(x)
    print(y)
    cmap = plt.get_cmap('RdYlBu')
    print(cmap)
    colors = cmap(x)  # find colors based on x
    print(colors)

if False:
    fig, ax = plt.subplots(1)

    max_concentration = 1  # maximum concentration at all levels
    num_levels = 20         # number of levels in the lake
    concentrations = np.linspace(0, max_concentration, num_levels)  # assign concentration at each level
    conc_color = concentrations / max_concentration  # have color vary from 0 to 1

    patches = []  # list to store polygons

    x = np.array([[1, 1],
                  [3, 1],
                  [3, 2],
                  [1, 2]])  # box of width 3 and thickness of 1

    for i in range(num_levels):
        x[:, 1] += 2
        polygon = Polygon(x, closed=True)
        patches.append(polygon)

    collection = PatchCollection(patches)
    ax.add_collection(collection)

    # set colors of polygons
    cmap = plt.get_cmap('ocean')
    colors = cmap(conc_color)  # scale concentrations on the colormap
    collection.set_color(colors)

    ax.autoscale_view()
    plt.show()

# Figuring out how to stack coordinates for visualization
if True:
    x1 = np.array([1, 2, 3, 4, 5])
    x1c = x1.reshape((x1.size,1)) # make a column vector
    x2 = np.array([6, 7, 8, 9, 10])
    x2c = x2.reshape((x2.size,1)) # make a column vector
    x3 = np.vstack((x1, x2))
    x3c = np.hstack((x1c, x2c))

    print(f'x1 is shape: {np.shape(x1)} and =\n{x1}')
    print(f'x1c is shape: {np.shape(x1c)} and =\n{x1c}')
    print(f'x2 is shape: {np.shape(x2)} and =\n{x2}')
    print(f'x1c is shape: {np.shape(x2c)} and =\n{x2c}')
    print(f'x3 is shape: {np.shape(x3)} and =\n{x3}')
    print(f'x3c is shape: {np.shape(x3c)} and =\n{x3c}')
print('end of the file')


# todo - come back to instructons at: https://matplotlib.org/3.2.1/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
