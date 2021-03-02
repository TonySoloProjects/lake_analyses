"""
Simulate/visualize water quality measurements using interactive figures
to inspect a lake's dissolved oxygen content and temperature.

Several figures include widget sliders to allow the user to select
a day of interest to for depth profiles of key water quality measurements
and interpolations.

Notes
-----
Author: Tony Held, tony.held@gmail.com
Created: 2020-05-14
1)  Lake volume versus elevation data may not exist to the zero depth (completely dry) point.
    Water features below the lowest elevation measured will not be considered.
    For the Loch Lomond analysis, this detail is likely trivial since this only ignores
    a volume of 1 acre-feet compared to the max volume of 8,646 acre-feet
    for other water bodies, you may need to extrapolate the lowest measured values to the lake bottom.
2)  The execution bottleneck of this code is the calculation of polygons using matplotlib.
    On my machine it takes 1 to 2 seconds to make these calcs which is up to 90% of total run time.
"""

import pickle

# Import numerical array data types and plotting libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider  # import the Slider widget
from scipy import interpolate

from tony_functions import MyTimer


class LakeMeasurements:
    """Class to simulate a lake to characterize its geometry and its water quality properties.

    Attributes
    ----------
    bath : Bathymetry
        Bathymetric information of water body including volume as a function of surface elevation
    wse_measured : WaterSurfaceElevations
        Time series of measured water surface elevations
    depth_meas : DepthMeasurements
        Time series of water quality data measured below surface
    """

    # Constants
    SQ_FT_PER_ACRE = 43560  # number of square feet in an acre
    L_PER_ACRE_FT = 1233481.85532  # number of liters per acre foot
    MG_PER_KG = 1000000  # milligrams per kilogram
    KG_PER_TONNE = 1000  # kilograms in a metric ton (tonne)

    def __init__(self):
        """Initialize member variables"""
        self.bath = None            # Bathymetric depiction of Lake
        self.wse_measured = None    # Measured water surface elevation data
        self.depth_meas = None      # Water quality measurements made below surface

    def read_csv_files(self, lake_geometry_fn='input/elevation_volume_v2.csv',
                       water_elevation_fn='input/wse.csv',
                       depth_measurements_fn='input/measurements_chopped.csv'):
        """Initialize lake for analysis by reading relevant data from csv files.

        Parameters
        ----------
        lake_geometry_fn : str
            Filename of lake geometry information
        water_elevation_fn : str
            Filename of water surface elevation measurements
        depth_measurements_fn : str
            Filename of oxygen and temperature measurements made at differing depths
        """

        # Read in cvs files with bathymetry, wse, and measurements
        self.bath = Bathymetry(lake_geometry_fn)
        self.wse_measured = WaterSurfaceElevations()
        self.wse_measured.read_wse_from_file(water_elevation_fn, False)
        self.depth_meas = DepthMeasurements(depth_measurements_fn)

        # add a water series to bathymetry
        # this will append plotting information to wse
        self.bath.add_wse("2010 to 2014 WSE's", self.wse_measured)

        # add depth measurements to bathymetry
        self.depth_meas.prep_data(self.bath, self.wse_measured, plot_data=False)

    def driver(self):
        """Current routine to create & test program functionality."""

        # Visual check that indices are being properly determined
        # Bathymetry.plot_wse_indices(self.bath.elevation, self.wse_measured)
        # plt.show()

        # Compare the radius & square approaches to visualize a lake cross section.
        self.plot_lake_visualizations()
        plt.show()

        # Plot surveyed and interpolated wse along with depth measurement locations
        InterpFunctions.plot_sampling_info(self.wse_measured.np_wse_times,
                                           self.wse_measured.wse,
                                           self.depth_meas.wse_interpolated.np_wse_times,
                                           self.depth_meas.wse_interpolated.wse,
                                           self.depth_meas.np_times,
                                           self.depth_meas.elevation)
        plt.show()

        self.depth_meas.plot_profile_slider('polygon')
        self.depth_meas.plot_profile_slider('line')

        self.plot_total_lake_do()

        self.plot_contours()

        plt.show()

    def sum_across_layers(self, volume, quality):
        """Find the total amount of some quality of the lake by summing
        the product of that quality at each level by the amount of water in that layer.

        Parameters
        ----------
        volume - numpy.ndarray(M,N)
            The volume of water at each layer of the lake
        quality - numpy.ndarray(M,N)
            A measured scalar value at elevations associated with volume

        Returns
        -------
        sum_product - numpy.ndarray(N)
            Scalar quality for each time period summed across all lake levels

        Notes
        -------
        1)  M is the number of layers in lake
            N is the number of sampling times considered
        """

        product = volume * quality
        sum_product = np.sum(product, axis=0)
        # print(np.shape(volume), np.shape(quality), np.shape(product), np.shape(sum_product))
        return sum_product

    def plot_total_lake_do(self):
        """Plot the lake's total dissolved oxygen along with WSE information.

        Total dissolved oxygen (DO) for each sampling period is
        the product of DO and the volume in each layer summed across each layers.

        Returns
        -------
        fig : figure.Figure
            Figure containing axis ax1 and ax2
        ax1 : axes.Axes
            axis with dissolved oxygen time-series
        ax2 : axes.Axes
            axis with wse time-series

        Notes
        -----
        """
        # todo - consider additional calculations to compare spreadsheet analysis accuracy

        # Find the midpoints and shape of the matrices to store interpolations
        midpoints = self.depth_meas.wse_interpolated.m_mid_elevation  # (feet)
        num_row, num_col = np.shape(midpoints)
        # print(num_row, num_col)

        # Shortcuts to sampling times, wse, and DO
        # ----------------------------------------
        # Sample times of depth measurements
        times = self.depth_meas.wse_interpolated.np_wse_times
        # Water surface elevation interpolated at depth measurement times
        wse = self.depth_meas.wse_interpolated.wse
        vol = self.depth_meas.wse_interpolated.m_vol_tot_af  # (acre-feet)
        # total DO in lake
        lake_do_per_period = self.depth_meas.mass_d_o_per_period  # (tonnes)

        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        # Visualize total lake DO
        fig1, ax1 = plt.subplots(1)
        fig1.subplots_adjust(right=0.75)
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        # Offset the right spine of ax3.  The ticks and label have already been
        # placed on the right by twinx above.
        ax3.spines["right"].set_position(("axes", 1.2))
        # Having been created by twinx, par2 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.
        make_patch_spines_invisible(ax3)
        # Second, show the right spine.
        ax3.spines["right"].set_visible(True)


        ax1.plot(times, lake_do_per_period, 'b', label="DO")
        ax1.set_xlabel('Time of depth measurements')
        ax1.set_ylabel('Metric Tonnes of DO')
        ax1.set_title('Loch Lomond Dissolved Oxygen, WSE, & Volume')

        ax2.plot(times, wse, 'g', label="WSE")
        ax2.set_ylabel('Water Surface Elevation (feet)')

        ax3.plot(times, vol, 'k', label="Volume")
        ax3.set_ylabel('Lake Volume (acre-feet)')

        fig1.legend()
        fig1.tight_layout()  # otherwise the right y-label is slightly clipped

        # print(np.shape(lake_do_per_layer), np.shape(midpoints))
        # print(midpoints[:, 0])
        # print(lake_do_per_layer[:, 0])

    def plot_contours(self):
        """Create contour maps showing temperature and dissolved oxygen at depth measurement times."""
        # todo - compare these calculations with spreadsheet calcs and figures

        # Shortcuts to sampling times, wse, and layer midpoints
        np_times = self.depth_meas.wse_interpolated.np_wse_times
        wse = self.depth_meas.wse_interpolated.wse
        y = self.depth_meas.wse_interpolated.m_mid_elevation
        num_row, num_col = np.shape(y)

        # Expand the time vector into a matrix so dimensions of contour plotting variables agree
        x = self.depth_meas.wse_interpolated.np_wse_times
        x1 = np.tile(x, (num_row, 1))
        # print(num_row, num_col, np.shape(x), np.shape(x1))

        # Y limits are now hard coded to Loch Lomond, but can be generalized in future applications
        ylim = (420, 580)

        # Plot Temperature and Dissolved Oxygen Figures
        fig1, ax1 = plt.subplots(1)
        contour1 = ax1.contourf(x1, y, self.depth_meas.m_temperature, cmap='winter')
        ax1.plot(np_times, wse, 'r-', linewidth=3)
        ax1.set_ylim(ylim)
        ax1.set_ylabel('Elevation (ft)')
        ax1.set_xlabel('Time of Below Surface Measurements')
        ax1.set_title("Temperature (oC)")
        fig1.colorbar(contour1, ax=ax1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

        fig2, ax2 = plt.subplots(1)
        contour2 = ax2.contourf(x1, y, self.depth_meas.m_d_o, cmap='winter')
        ax2.plot(np_times, wse, 'r-', linewidth=3)
        ax2.set_ylim(ylim)
        ax2.set_ylabel('Elevation (ft)')
        ax2.set_xlabel('Time of Below Surface Measurements')
        ax2.set_title("Dissolved Oxygen (mg/L)")
        fig1.colorbar(contour2, ax=ax2)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
        #  plt.show()

    def _test_diagnostics(self):
        """Run various diagnostic tests to test key functionality"""
        self._test_lake_geometry()
        self._test_vector_calcs()
        self._test_profile_interpolation()

    def plot_lake_visualizations(self):
        """Visualization the lake with differing plotting approaches to see which one looks best.

        Notes
        -------
        1) This routine was used in development, but now is primarily designed to create 2 figures
        2) Figure 1. Demonstrates the difference between various ways to visualize a lake cross-section
        3) Figure 2. Show the lake geometry when full along with elevation labeling (spillway, deadpool, etc)"""

        # Find the polygon's associated with representing the cross section of the lake
        # as either cylindrical or square with left and centered alignments
        verts_radial_center, patches_radial_center = Bathymetry.calc_polygons(
            self.bath.elevation, self.bath.radii, alignment='center')
        verts_radial_left, patches_radial_left = Bathymetry.calc_polygons(
            self.bath.elevation, self.bath.radii, alignment='left')
        verts_linear_center, patches_linear_center = Bathymetry.calc_polygons(
            self.bath.elevation, self.bath.vol_inc_af_per_z, alignment='center')
        verts_linear_left, patches_linear_left = Bathymetry.calc_polygons(
            self.bath.elevation, self.bath.vol_inc_af_per_z, alignment='left')

        # Each layer can be color coded based on a scalar value
        # For these plots, the color will be associated with the amount of water in each layer
        water_per_layer = self.bath.vol_inc_af  # in acre-feet, used to colormap layers

        # Compare stacked cylinder to stacked rectangle visualization
        fig1, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
        Bathymetry.plot_polygons(patches_radial_center, ax1, values=water_per_layer)
        ax1.set_ylabel('Elevation (ft)')
        ax1.set_xlabel('Radii of disc (ft)')
        ax1.set_title("Layers as cylinders")
        Bathymetry.plot_polygons(patches_linear_center, ax2, values=water_per_layer)
        ax2.set_title("Layers as rectangles")
        ax2.set_xlabel('Normalized Surface Area')
        fig1.suptitle('Comparison of Lake Visualization')

        # Compare water cross section with vertical profile of surveyed wse measures
        fig3, [fig3_ax1, fig3_ax2] = plt.subplots(1, 2, sharey='row', gridspec_kw={'width_ratios': [5, 2]})
        self.bath.plot_polygons(patches_linear_left, fig3_ax1, values=water_per_layer)
        self.plot_profile(fig3_ax2)
        fig3.suptitle('Cross section of Lake and Surveyed WSE')
        fig3_ax1.set_ylabel('Elevation (ft)')
        fig3_ax1.set_xlabel('Normalized Cross-Sectional Area')
        fig3_ax2.set_xlabel('Volume of lake (acre-feet)')

        # seeing what the wse and polygons look like overlapped
        # fig2, fig2_ax1 = plt.subplots(1)
        # self._plot_polygons(self.patches4, fig2_ax1)
        # self._plot_profile(fig2_ax1)

        #  plt.show()

    def plot_profile(self, ax=None):
        """Plot stage elevation (feet) as a function of storage (cubic feet).

        Parameters
        ----------
        ax : axes.Axes
            Axis to plot polygons to, if not supplied, a new figure and axis are created

        Returns
        ----------
        ax : axes.Axes
            Axis used in polygon plots

        Notes
        ------
        1)  Profile data is currently hard coded for Loch Lomond, but it can be
            updated to be dynamic for future lake analysis.
        """

        if ax is None:
            fig, ax = plt.subplots(1)

        # Plot the surveyed relationship between water surface elevation and lake storage
        ax.plot(self.bath.vol_af, self.bath.elevation)

        # Plot the key elevations specific to Loch Lomond for context

        # format: name of level, elevation, % of volume below, total volume below
        key_elevations = [
            ['mud level', 460, 1, 54],
            ['original dead pool', 470, 2, 141],
            ['current dead pool', 490, 7, 625],
            ['managed reserve', 534, 36, 3088],
            ['spillway', 577.51, 100, 8646]]

        for key_elv in key_elevations:
            name = key_elv[0]
            elev = key_elv[1]
            vol = key_elv[3]
            label_vol = 4000  # x coordinate of where label line ends
            if vol > label_vol:
                align = 'right'
            else:
                align = 'left'

            ax.hlines(elev, vol, label_vol, colors='r')
            ax.text(label_vol, elev, name, ha=align, va='center')

    def _test_lake_geometry(self):
        """Test geometry functions by calculating cumulative volume, ect
           by exporting to csv and then checking in spreadsheet.
        """

        # export geometry to cvs file for inspection
        np.savetxt("output/radii_v2.csv", self.bath.radii, delimiter=",")
        np.savetxt("output/thickness_v2.csv", self.bath.thickness, delimiter=",")

    def _test_profile_interpolation(self):
        """Test ways to interpolate measurement profile based on measurements at differing depths.

         Initial testing used the full lake geometry with the first dissolved oxygen profile
         as a case study.
         """

        # load in 1st measurement and save the 5 measurements made at different depths
        measurements_fn = 'input/measurements_chopped.csv'
        measurement_df2 = pd.read_csv(measurements_fn, index_col=0, parse_dates=True)
        # Sort 1st by date ascending, then by elevation ascending
        measurement_df2.sort_values(by=['Date', 'Elevation (ft AMSL)'], ascending=[True, True], inplace=True)

        np_times2 = measurement_df2.index.to_numpy(dtype=np.datetime64)
        measurements2 = measurement_df2.to_numpy()
        elev2 = measurements2[:, 0]
        temperature2 = measurements2[:, 1]
        d_o2 = measurements2[:, 2]
        my_e = elev2[0:5]
        my_t = temperature2[0:5]
        my_d_o = d_o2[0:5]
        # print(my_e, my_t, my_d_o)

        elev_midpoints = self.bath.mid_elevation
        # print(np.shape(elev_midpoints), elev_midpoints)

        # interpolate, anything above the top layer measurement is set to the top layer measurement
        # interpolate, anything below the bottom layer measurement is set to the bottom layer measurement
        bottom_layer = my_d_o[0]
        top_layer = my_d_o[-1]
        # print(bottom_layer, top_layer)

        f = interpolate.interp1d(my_e, my_d_o, fill_value=(bottom_layer, top_layer), bounds_error=False)
        new_d_o = f(elev_midpoints)

        fig, ax = plt.subplots(1)

        ax.plot(new_d_o, elev_midpoints, 'bx-', my_d_o, my_e, 'ro')
        #  plt.show()

    @staticmethod
    def _test_vector_calcs():
        """Test how numpy implements matrix calculations and explore some practice calculations."""

        a = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        # a = np.array([[1], [2], [3], [4], [5]])
        b = np.tile(a, (1, 2))
        print(f'type(a), {type(a)}, a.size, {a.size}, a.shape, {a.shape}\n{a}')
        print(f'type(b), {type(b)}, b.size, {b.size}, b.shape, {b.shape}\n{b}')
        # print(f'type(c), {type(c)}, c.size, {c.size}, c.shape, {c.shape}\n{c}')
        # print(f'type(self.wse), {type(self.wse)}, self.wse.size, {self.wse.size}, self.wse.shape, {self.wse.shape}\n{self.wse}')

        d = np.array([[1],
                      [2],
                      [3]
                      [4]
                      [5]])


class Bathymetry:
    """Class to store bathymetric information about a waterbody.

    Attributes
    ----------
    elevation : numpy.ndarray(M)
        elevation associated with survey data (feet).
    vol_af : numpy.ndarray(M)
        Total volume associated with elevation measurement (acre-feet).
    vol_cf : numpy.ndarray(M)
        Total volume associated with elevation measurement (cubic-feet).
    vol_cum_perc : numpy.ndarray(M)
        Cumulative percent water associated with elevation measurement compared to full lake.
    mid_elevation : numpy.ndarray(M-1)
        Midpoint elevation of each layer (feet).
    thickness : numpy.ndarray(M-1)
        Midpoint elevation of each layer (feet).
    vol_inc_af : numpy.ndarray(M-1)
        Incremental volume in each layer (acre-feet).
    vol_inc_cf : numpy.ndarray(M-1)
        Incremental volume in each layer (cubic-feet).
    vol_perc : numpy.ndarray(M-1)
        Percent of max lake volume in each layer
    radii : numpy.ndarray(M-1)
        Radius of each layer if lake simulated as stacked cylinders
    vol_inc_af_per_z : numpy.ndarray(M-1)
        Length of each layer if lake is to be visualized
        in a way that plot area is proportional to water volume
    wse_dict
        # used to store time series of water surface elevation

    Notes
    ----------
    1)  M number of elevations associated with M-1 layers
    2)  elevations and layers are ordered bottom to top.  For instance,
        elevation(i) is the bottom of a layer and elevation(i+1) is its top
    3)  The elevations and water content in the bathymetry file is from surveys
        and is not time dependent
    4)  Water surface elevations stored in wse_dict are from measurement time series
    """

    def __init__(self, lake_geometry_fn):

        # Initialize attributes
        self.elevation = None
        self.vol_af = None
        self.vol_cf = None
        self.vol_cum_perc = None
        self.mid_elevation = None
        self.thickness = None
        self.vol_inc_af = None
        self.vol_inc_cf = None
        self.vol_perc = None
        self.radii = None
        self.vol_inc_af_per_z = None
        self.wse_dict = {}  # use to store water surface elevation information

        # Read bathymetric data from file and initialize object attributes
        elevation, vol_af = Bathymetry._read_geometry_from_file(lake_geometry_fn)
        self._initialize_geometry(elevation, vol_af)

    @staticmethod
    def _read_geometry_from_file(lake_geometry_fn):
        """Read in lake geometry information from a file that contains
         water surface elevations versus lake volumes.

        Parameters
        ----------
        lake_geometry_fn : str
            Filename of surface elevation versus lake volume in csv format.
            First line is Elevation (ft), Volume (ac-ft).
            Subsequent lines have numerical data

        Returns
        -------
        elevation : numpy.ndarray(M)
            elevation associated with survey data (feet).
        vol_af : numpy.ndarray(M)
            Total volume associated with elevation measurement (acre-feet).
        """

        # Read in bathymetry, sort in panda, save to numpy arrays for further use
        lake_data_df = pd.read_csv(lake_geometry_fn)
        # Sort by elevation ascending so that lake indexing is from the bottom up
        lake_data_df.sort_values(by=['Elevation (ft)'], ascending=True, inplace=True)

        # Extract dataframe information as numpy arrays
        lake_data = lake_data_df.to_numpy()
        elevation = lake_data[:, 0]  # elevation (feet)
        vol_af = lake_data[:, 1]  # total volume in (acre-feet)

        return elevation, vol_af

    def _initialize_geometry(self, elevation, vol_af):
        """Initialize bathymetric object attributes based on elevation and water volume information.

        Parameters
        ----------
        elevation : numpy.ndarray(M)
            elevation associated with survey data (feet).
        vol_af : numpy.ndarray(M)
            Total volume associated with elevation measurement (acre-feet).

        Notes
        ----------
        1) Sets the following attributes:
            elevation,
            vol_af,
            vol_cf,
            max_volume_cf,
            vol_cum_perc,
            mid_elevation,
            thickness,
            vol_inc_af,
            vol_inc_cf,
            vol_perc,
            radii,
            vol_inc_af_per_z
        """

        # ------------------------------------------------------------
        # Variables that are the same length of the # of elevations in original data
        # ------------------------------------------------------------
        self.elevation = elevation  # elevation (feet)
        self.vol_af = vol_af  # total volume in (acre-feet)
        self.vol_cf = self.vol_af * LakeMeasurements.SQ_FT_PER_ACRE  # total volume in (cubic-feet)
        max_volume_cf = self.vol_cf[-1]  # volume when full at spillway
        self.vol_cum_perc = self.vol_cf / max_volume_cf  # Cumulative percent of water at elevation

        # ------------------------------------------------------------
        # Data associated with each layer
        # The number of layers is one member shorter than the total elevations
        # Layer[0] is the bottom most layer defined by
        # elevation[0] - the bottom of Layer[0]
        # elevation[1] - the top of Layer[0]
        # ------------------------------------------------------------
        bottoms = self.elevation[:-1]
        tops = self.elevation[1:]
        self.mid_elevation = (tops + bottoms) / 2  # midpoint elevation of the layer
        self.thickness = np.diff(self.elevation)  # thickness of layer (feet)
        self.vol_inc_af = np.diff(self.vol_af)  # incremental volume in each layer (acre-feet)
        self.vol_inc_cf = np.diff(self.vol_cf)  # incremental volume in each layer (cubic-feet)
        self.vol_perc = self.vol_inc_cf / max_volume_cf  # percent of max lake volume in each layer

        # ---------------------------------------------------------------------------
        # Calculate the geometric properties for each layer to assist in visualization
        # ---------------------------------------------------------------------------
        # Two types of projects are considered.
        # Each type of projection will highlight and/or distort how data are presented

        # Representing each layer as a disk shown in cross section will make the smaller volume layers
        # appear bigger (because the size of the disk is related to the square of the radius)
        # Thus a layer of radius 2 will have 4 times the volume of a layer or raidus 1

        # The radius (feet) of each layer assuming it is a cylinder
        # V = pi*r^2*h
        # r = (V/(pi*h))^1/2
        self.radii = (self.vol_inc_cf / (np.pi * self.thickness)) ** 0.5

        # vol_inc_af_per_z is the incremental acre-feet per foot of vertical in the layer
        # Using this variable to visualizing the lake as stacked polygons more intuitive because
        # the area of the polygon will be proportional to the volume of water in the layer
        # V = length * width * height
        # width = V/(l*height)
        # We will be plotting a cross section, so length in the unseen axis is set to (1)
        # The height of each layer varies, so the incremental volume per layer is normalized by layer thickness
        self.vol_inc_af_per_z = self.vol_inc_af / self.thickness

    @staticmethod
    def calc_polygons(elevation, width, alignment='center'):
        """Calculate the polygons required to visualize a water body in matplotlib
        for a single set (vector) of elevations and widths.

        Parameters
        ----------
        elevation : numpy.ndarray(M)
            elevation associated with survey data (feet).
        width : numpy.ndarray(M)
            Horizontal dimension of polygon (either radius or linear width)
        alignment : {'center', 'left'}
            Determines how polygons will be aligned in subsequent plotting routines.

        Returns
        -------
        verts : numpy.ndarray(M, 4, 2)
            Vertices associated with 4 coordinate points (x,y) locations
        patches : list of polygons
            List storing polygons for that can be visualized in matplotlib.

        Notes
        -----
        The coordinates system for a lake's 2-d cross section are shown below.
        Each layer will appear as a rectangle with four corners.
        The coordinates of each corner is determined as follows:

           (x1, y1)  -------|--------- (y4, x4)
               |            |              |
           (x2, y2)  -------|--------- (y3, x3)
                            CL
        Where:
           CL - is the center-line of the disc (x=0 from y=0 to layer thickness)
           (y2-y1) = (y4-y3) = layer thickness
        If the polygon is centered
           x1 = x2= - width
           x3 = x4= + width
        If the polygon is left aligned
           x1 = x2= 0
           x3 = x4= + width
        """

        # Calculate 2-d cross section coordinates
        y1 = y4 = elevation[1:]  # Top of disc (excludes bottommost elevation)
        y2 = y3 = elevation[:-1]  # Bottom of each disc (excludes topmost elevation)
        x3 = x4 = + width

        if alignment == 'center':
            x1 = x2 = - width
        elif alignment == 'left':
            x1 = x2 = np.zeros(np.shape(width))
        else:
            print(f'Unknown alignment type selected: {alignment}')

        # Combine individual coordinates into a matrix to speed up polygon creation
        # Each coordinate component (e.g., x1) is a vector represented as a row
        # use vstack to combine data into a 2-d matrix where each row
        # contains the 8 data points needed to describe the polygon in the order
        # (x1, y1), (x2, y2), (x3, y3), (x4, y4) for each disc

        coordinates = np.vstack((x1, y1, x2, y2, x3, y3, x4, y4)).transpose()
        (coord_rows, coord_cols) = np.shape(coordinates)

        # export coordinates to cvs file for inspection
        # np.savetxt("output/coords.csv", coordinates, delimiter=",")

        # Reshape the coordinates so that is is a 3d matrix which stores
        # the coordinates in [z, 4, 2] values suitable for plotting functions
        verts = np.reshape(coordinates, (-1, 4, 2))

        # (vert_groups, vert_rows, vert_cols) = np.shape(verts)
        # print(f'coordinate rows={coord_rows} col={coord_cols}')
        # print(f'first row of coordinates is: {coordinates[0, :]}')
        # print(f'verts groups={vert_groups} rows={vert_rows} col={vert_cols}')
        # print(f'first groups of  is: {verts[0,:,:]}')

        # Create a list of polygons from coordinate data
        patches = []  # list to store polygons for plotting lake
        for vert in verts:
            polygon = Polygon(vert, closed=True)  # This routine can be up to 90% total runtime
            patches.append(polygon)
        return verts, patches

    @staticmethod
    def calc_multi_polygons(elevation, width, alignment='center'):
        """Calculate the polygons required to visualize a water body in matplotlib
        for multiple sets (matrix) of elevations and widths.  See calc_polygons for details.

        Parameters
        ----------
        elevation : numpy.ndarray(M x N)
            elevation associated with survey data (feet).
        width : numpy.ndarray(M x N)
            Horizontal dimension of polygon (either radius or linear width)
        alignment : {'center', 'left'}
            Determines how polygons will be aligned in subsequent plotting routines.

        Returns
        -------
        verts : list
            List of length N where each member is a matrix of vertices of size/type
            numpy.ndarray(M, 4, 2) associated with each layer.
        patches : list
            List of length N where each member is a polygon patch that can be used in matplotlib"""

        verts = []
        patches = []
        num_rows, num_col = np.shape(elevation)

        # loop through each time series and calc associated verts and patches
        for j in range(num_col):
            v, p = Bathymetry.calc_polygons(elevation[:, j], width[:, j], alignment)
            verts.append(v)
            patches.append(p)

        return verts, patches

    @staticmethod
    def plot_polygons(patches, ax=None, values=None, max_value=None):
        """Visualize the lake based on polygon patches created by calc_polygons method.

        Parameters
        ----------
        patches : list of length M
            List of polygons to represent each lake level.
        ax : axes.Axes
            Axis to plot polygons to, if None a new figure and axis are created
        values : numpy.ndarray(M)
            Value associated with each patch member, e.g., the temperature at each level.
            These will be used to colormap the patches
            If None a random data are used for color plotting purposes.
        max_value : (int or float)
            Maximum maximum value considered for labeling purposes.  Use of this parameter
            allows for the same colorbar range to be used if multiple patch plots are considered.

        Returns
        -------
        ax : axes.Axes
            axis used for plotting patches.
        """

        if ax is None:
            fig, ax = plt.subplots(1)

        # Transform the measured values to plot_colors that range from 0 and 1
        # If no values are associated with each layer then assign them random values
        if values is None:
            values = np.random.random(len(patches))
        if max_value is None:
            max_value = np.max(values)
        plot_color = values / max_value

        # Store the polygons as a patch collection because this will render faster and
        # make it easier to apply a colormap to the polygons
        collection = PatchCollection(patches)

        # Set colormap and determine associated colors for each patch
        cmap = plt.get_cmap('Blues')
        colors = cmap(plot_color)

        # Assign color to polygon patches
        collection.set_color(colors)
        collection.set_edgecolor('k')

        # Display the polygons to the axis
        ax.add_collection(collection)
        ax.autoscale_view()
        #  plt.show()

        return ax

    def add_wse(self, memo, my_wse):
        """Add time and water surface elevation measurements to a lake which
         is stored as a dictionary in the Bathymetry object.

        Parameters
        ----------
        memo : str
            Descriptor used in dictionary to describe this wse dataset.
        my_wse : WaterSurfaceElevations
            wse object being added to bathymetry's dictionary.
            The wse object will be augmented with additional information
            e.g. (elevation information) to assist in subsequent plotting.
        """

        # Add a new water surface elevation key to the bathymetry object
        self.wse_dict[memo] = my_wse

        # Find the index that a measured wse should be associated with the
        # survey of water elevations versus surface area stored in the bathymetry object
        for z in my_wse.wse:
            i = Bathymetry.find_elevation_index(z, self.elevation)
            my_wse.top_index.append(i)

        # Append additional attributes to a wse object, based on lake bathymetry, to simplify plotting routines
        WaterSurfaceElevations.append_bathymetry_to_wse(my_wse, self.elevation, self.thickness, self.vol_inc_af)

    @staticmethod
    def find_elevation_index(my_z, elevations):
        """Find the index of the top of the layer associated with with a wse measurement.

        The index value will be used later to reset the topmost elevation for visualization purposes and to
        adjust water volumes at the topmost layer and zero out water in layers above the measured wse.

        Parameters
        ----------
        my_z : (int or float)
            elevation (feet) of a measurement
        elevations : numpy.ndarray(M)
            Vector of elevations associated with survey of wse vs lake volume.

        Returns
        -------
        index : integer
            index of the top of the layer associated with with a wse measurement

        Notes
        -----
        1) elevations[i] - bottom of the layer of bathymetry data when full
        2) elevations[i+1] - top of the layer of bathymetry data when full
        3) for example:
            elevations[71] = 577.51
            elevations[70] = 576
            my_z = 577
            results in an index = 71"""

        # Find the topmost index associated with measured wse
        # Initialized to -1, and should be updated to be a positive # below
        # If a negative number slips through,
        #   then it will be an array indexing error and stop the program
        index = -1

        # Surface measurements above the max lake height are allowed
        # as long as they are 0.25 feet (3 inches) from max elevation from bathymetry data
        max_elevation_tolerance = 0.25

        # Precision considered for the comparison of two elevations.
        # It is not currently implemented, but consider using it
        #   if there are machine number rounding issues
        elevation_precision = 0.01

        # Find the min, max elevations, and maximum possible index value
        min_z = elevations[0]  # bottom of the lowest layer
        max_z = elevations[-1]  # top of the highest layer
        max_index = np.size(elevations)  # max index is the topmost elevation in the survey

        # Ensure the measured elevation is within the range of possible values.
        # If the measurement is outside that range,
        #   but still within max_elevation_tolerance of an acceptable value,
        #   correct it to be within the acceptable range.
        # Send warnings to a log file to keep the debug window from getting cluttered.
        with open('warnings_log.txt', 'a') as file_object:
            if my_z > max_z or my_z < min_z:
                file_object.write('**Specified elevation outside the modeled range**\n')
                file_object.write(f'\trequested elevation = {my_z}, min z {min_z}, max z {max_z}\n')
                if (my_z - max_z) < max_elevation_tolerance:
                    file_object.write(
                        f'\tsince measured elevation is less than 0.25 feet above of surveyed max elevation, '
                        f'the measured data will be reset to max surface elevation of {max_z}\n')
                    my_z = max_z
                else:
                    return  # measurement is too far above max allowable measurement

        # Find the index for the top of the trimmed layer.
        # Sort from the top to the bottom for increased speed (since the wse is usually near the top)
        # First find the index of the bottom layer associated with the measurement.
        # Once you find that, the top layer is one index above.

        for i, z in reversed(list(enumerate(elevations))):
            # print(i, z)
            if my_z > z:
                index = i + 1
                break
        # print(f'{elev} is between elevation {self.elevation[index-1]} and {self.elevation[index]} at index {index}')

        return index

    @staticmethod
    def plot_wse_indices(elevations, my_wse):
        """Present the wse as a time series with an overlay of the lake layers.

        This method was created to visually check that the top indices are calculated
        correctly, and has little usage other than debugging.

        Parameters
        ----------
        elevations : numpy.ndarray(M)
            Fixed elevations from bathymetric survey data.
        wse : WaterSurfaceElevations
            Water surface elevations to be evaluated
        """

        # Find times and surface elevations
        wse_times = my_wse.np_wse_times
        wse_elevations = my_wse.wse

        # Plot wse time series
        fig, myax = plt.subplots(1)
        myax.plot(wse_times, wse_elevations, 'bx--', markersize=3)
        myax.set_title("Comparing measured to interpolated WSE's")
        myax.set_ylabel('Elevation (ft)')
        myax.set_xlabel('Time of measurement or interpolation')

        # Plot the surveyed elevations as horizontal lines
        myax.hlines(elevations, wse_times[0], wse_times[-1], 'g')

        # Plot the calculated index as text next to each measurement to visually check results
        for ts, my_wse, ti in zip(wse_times, wse_elevations, my_wse.top_index):
            myax.text(ts, my_wse, ti, ha='left', va='top')


class WaterSurfaceElevations:
    """Store a time series of water surface elevations for a lake and
    calculate plotting information required to visualize data.

    Attributes
    ----------
    water_elevation_fn : str
        Name of file if data are read from a csv text file
    np_wse_times : numpy.ndarray(M) of type np.datetime64
        Date and time of measurement.
    wse : numpy.ndarray(M)
        Measured water surface elevations (feet)
    num_measurements :
        Number of wse measurements
    top_index : List of integers
        Indexing information that is useful if the wse's are mapped to bathymetric information.
        Each value is the index of the bathymetric elevations associated with the wse.
    m_elevations : numpy.ndarray(M x N)
        Elevation of simulated water body (feet)
    m_thickness_full : numpy.ndarray(M-1 x N)
        Thickness of each layer if the water body is full (feet)
    m_vol_inc_af_full : numpy.ndarray(M-1 x N)
        Incremental water volume if each layer each layer if the water body is full (acre-feet)
    m_vol_inc_af : numpy.ndarray(M-1 x N)
        Incremental water volume if each layer each layer after accounting for it not being full (acre-feet)
    m_vol_tot_af = numpy.ndarray(N)
        Total water volume for each time period after accounting for it not being full (acre-feet)
    m_vol_inc_af_per_z : numpy.ndarray(M-1 x N)
        Normalized volume in each simulated layer suitable for plotting routines.
    m_mid_elevation : numpy.ndarray(M-1 x N)
        Elevation of midpoint of each layer of simulated water body (feet)
    m_ratio : numpy.ndarray(M-1 x N)
        Ratio of simulated water body at a given layer to its full lake level equivalent
    verts : list
        List of length N where each member is a matrix of vertices of size/type
        numpy.ndarray(M, 4, 2) associated with each layer.
    patches : list
        List of length N where each member is a polygon patch that can be used in matplotlib

    Notes
    -----
    1)  Attributes prefixed with m_, verts, and patches are
        appended to the wse by the method append_bathymetry_to_wse.
    2)  Appended attributes combine bathymetric data to a wse time series data to create various 2-d matrices.
    3)  M is the # of rows in each appended matrix is the number of elevations in the full lake survey.
    4)  N is the # of columns in each appended matrix is the number of wse time series measurements.
    5)  The numpy arrays are initialized as if the lake was full for each water surface measurement.
        The arrays are subsequently trimmed/prorated to account for the fact that the
        wse measurements may be made at times other than the lake being completely full.
    """

    def __init__(self):
        self.water_elevation_fn = None
        self.np_wse_times = None
        self.wse = None
        self.num_measurements = None
        self.top_index = []
        self.m_elevations = None
        self.m_thickness_full = None
        self.m_vol_inc_af_full = None
        self.m_vol_inc_af = None
        self.m_vol_tot_af = None
        self.m_vol_inc_af_per_z = None
        self.m_mid_elevation = None
        self.m_ratio = None

        self.verts = None
        self.patches = None

    def set_wse_and_time(self, np_wse_times, wse):
        """Set wse based on time data and water surface elevation measurements.

        Parameters
        ----------
        np_wse_times : numpy.ndarray(M) of type np.datetime64
            Date and time of measurement.
        wse : numpy.ndarray(M)
            Measured water surface elevations (feet)
        """

        self.np_wse_times = np.copy(np_wse_times)
        self.wse = np.copy(wse)
        self.num_measurements = np.size(self.np_wse_times)

    def read_wse_from_file(self, water_elevation_fn, plot_data=False):
        """Read water surface elevations (wse) from a text file.

        Parameters
        ----------
        water_elevation_fn : str
            Filename associated with water surface elevations information.
        plot_data : {True, False}
            Boolean to determine if data are to be plotted
        """

        # Read data and then sort it by time stamp (ascending)
        wse_df = pd.read_csv(water_elevation_fn, index_col=0, parse_dates=True)
        wse_df.sort_values(by=['Date'], ascending=True, inplace=True)

        # Save the data to attribute variables
        self.water_elevation_fn = water_elevation_fn
        self.np_wse_times = wse_df.index.to_numpy(dtype=np.datetime64)
        self.wse = wse_df.to_numpy().flatten()
        self.num_measurements = np.size(self.wse)

        if plot_data:
            my_ax = wse_df.plot(marker='o', markerfacecolor='r', )
            my_ax.set_title(f'{self.num_measurements} Measured Water Surface Elevations')
            my_ax.set_ylabel('Elevation (feet)')
            #  plt.show()

        # debugging statements below
        # print(type(np_times), np.shape(np_times), np_times)
        # print(type(wse), np.shape(wse), wse)
        # plt.plot(np_times, wse, marker='o', markerfacecolor='r')

    @staticmethod
    def append_bathymetry_to_wse(my_wse, elevation, thickness, vol_inc_af):
        """Append attributes to water surface elevation time series based on bathymetric data to
        allow for easy calculation of water quality values and to simplify plotting.

        The appended attributes are all prefixed with m_ are are described in the class definition.

        Parameters
        ----------
        my_wse : WaterSurfaceElevations
            Time series of wse measurements to be visualized/simulated.
            numpy arrays simulating the lake will be appended to this wse object
        elevation : numpy.ndarray(M)
            Elevations from bathymetric survey of full lake (feet)
        thickness : numpy.ndarray(M-1)
            Thickness of each full layer from bathymetric survey (feet)
        vol_inc_af : numpy.ndarray(M-1)
            Incremental volume in each layer from bathymetric survey (acre-feet)
        """

        # First consider the lake to be completely full, and then modify the
        # m_ matrices to account for the measured/simulated wse

        # Initializing the matrices as full is accomplished by
        # expanding the bathymetric elevation, thickness, and incremental volume variables
        # with as many columns as there are in wse measurements

        # The shape of each matrix is i, j
        # where i is the number of vertical levels in the bathymetric data and
        # j is the number of time series in the wse data

        my_wse.m_elevations = np.tile(elevation, (my_wse.num_measurements, 1)).transpose()
        # print(my_wse.m_elevations.shape)

        my_wse.m_thickness_full = \
            np.tile(thickness, (my_wse.num_measurements, 1)).transpose()  # layer thickness in feet
        my_wse.m_vol_inc_af_full = \
            np.tile(vol_inc_af,
                    (my_wse.num_measurements, 1)).transpose()  # incremental volume in each full layer (acre-feet)

        # Adjust the wse appended matrices to account for the fact
        #   the lake may not be full for each wse time period
        WaterSurfaceElevations.adjust_appended_wse(my_wse)

    @staticmethod
    def adjust_appended_wse(my_wse):
        """Adjust the m_ matrices appended to the wse object
        to account for the fact that the lake may not be full at a given wse.

         This adjustment is achieved by creating a matrix of elevations with the same number
         of rows as the bathymetric full lake conditions with a column associated with each wse element.
         Each column is then adjusted so that the actual water surface elevation is the max elevation considered.
         All elevations above the current water line are set to the current water line so that
         no information is visualized by the plotting routines above the wse.
         m_ matrices calculating the thickness, volume of water in, etc each layer are created.
         The length of the full lake data may be longer than the length of the vector for a non-full wse.
         The top_index is the uppermost value considered for non-full information.
         Values for indices greater than top_index are zero'd out since there is no water/data above the wse.

        Parameters
        ----------
        my_wse : WaterSurfaceElevations
            Time series of wse measurements to be visualized/simulated.
            numpy arrays simulating the lake will be appended to this wse object
        """

        # Minimum elevation difference (in feet) that is considered significant
        # Any calculated difference less than this will be set to zero to avoid machine number issues
        min_elevation_difference = 0.001

        # layer information preceded with an m_ is a 2-d matrix
        # where each column is elevation information for a single water surface elevation

        # Set all elevations that are above the top elevation = current measured elevation
        wse_measurements = range(my_wse.num_measurements)
        for j in wse_measurements:  # loop through each wse in time series
            i = my_wse.top_index[j]  # find the topmost elevation associated with wse
            z = my_wse.wse[j]  # The measured wse elevation
            # print(f'i {i}, j {j}, z {z}')
            # print(my_wse.m_elevations[i:, j])
            my_wse.m_elevations[i:, j] = z  # Set all elevations above the wse line = wse line
            # print(my_wse.m_elevations[i:, j])

        # Find the midpoint of of each layer
        bottoms = my_wse.m_elevations[:-1, :]
        tops = my_wse.m_elevations[1:, :]
        my_wse.m_mid_elevation = (tops + bottoms) / 2  # midpoint elevation of the layer

        # Find the new thickness in each layer based on time series of wse measurements
        # The thickness of the layer will be used subsequently to prorate measured water volumes to full water volumes
        # If the thickness of a layer is =0, then it has no water and will be zeroed out in most analyses
        my_wse.m_thickness = np.diff(my_wse.m_elevations, axis=0)  # thickness of layer (feet)

        # set any thickness less than 0.0001 feet equal to zero to avoid machine number issues
        my_wse.m_thickness[np.absolute(my_wse.m_thickness) < min_elevation_difference] = 0

        # find the ratio of volume in the measured layer compared to full lake volume
        my_wse.m_ratio = my_wse.m_thickness / my_wse.m_thickness_full

        # print(self.m_elevations.shape)
        # print(self.m_thickness.shape)
        # print(self.m_ratio.shape)

        # find the incremental volume in each layer for each wse (acre-feet)
        my_wse.m_vol_inc_af = my_wse.m_vol_inc_af_full * my_wse.m_ratio

        # find the total volume in lake per time period (acre-feet)
        my_wse.m_vol_tot_af = np.sum(my_wse.m_vol_inc_af, axis=0)

        # Find the incremental volume normalized by layer depth in each layer for each wse (acre-feet/feet)
        # to be able to visualize the layer data in the polygon plotting routines
        # It is initialized to 0 and then only the non-zero thicknesses are considered to avoid
        # divide by zero issues
        my_wse.m_vol_inc_af_per_z = np.zeros(my_wse.m_thickness.shape)  # initialize all values to zero
        non_zeros = np.absolute(
            my_wse.m_thickness) > min_elevation_difference  # find where the layer is not of zero thickness
        my_wse.m_vol_inc_af_per_z[non_zeros] = my_wse.m_vol_inc_af[non_zeros] / (
        my_wse.m_thickness[non_zeros])  # perform normalizing calcs for non-zero layers

        # create patches for each lake profile to speed up visualizations later
        # my_timer = MyTimer(f"Calculating Verts for {my_wse.num_measurements} wse measurements")
        my_wse.verts, my_wse.patches = \
            Bathymetry.calc_multi_polygons(my_wse.m_elevations, my_wse.m_vol_inc_af_per_z, alignment='left')
        # my_timer.toc()


class DepthMeasurements:
    """Store water quality measurements made at different times and depths and interpolate them
    to facilitate calculations and for plotting purposes.

    Attributes
    ----------
    depth_measurements_fn : str
        File name with water quality measurements made at depth
    measurement_df : pandas.DataFrame
        water quality measurement data as a panda data structure
    np_times : numpy.ndarray(N_total) of type np.datetime64
        Date/times that depth measurements were made
    np_times_unique : numpy.ndarray(N_unique) of type np.datetime64
        Unique Date/times that depth measurements were made
    measurements : numpy.ndarray(N_total, 3)
        numpy representation of the pandas.DataFrame which stores the
        elevation, temperature, and dissolved oxygen content for each measurement time
    elevation : numpy.ndarray(N_total)
        Elevations associated with each row of data in the measurement file (feet)
    temperature : numpy.ndarray(N_total)
        Temperature associated with each row of data in the measurement file (oC)
    d_o : numpy.ndarray(N_total)
        Dissolved oxygen associated with each row of data in the measurement file (mg/L)
    wse_interpolated : WaterSurfaceElevations
        Water surface elevations for the N_unique time periods
    jagged_elevations : Nested List of floats
        Elevations associated with a unique time measurement (feet)
    jagged_temperature : Nested List of floats
        Temperatures associated with a unique time measurement (oC)
    jagged_d_o : Nested List of floats
        Dissolved oxygen associated with a unique time measurement (mg/L)
    m_temperature : numpy.ndarray(M, N_unique)
        Interpolated temperature at lake layer midpoints for each unique time measurement (oC)
    m_d_o : numpy.ndarray(M, N_unique)
        Interpolated dissolved oxygen at lake layer midpoints for each unique time measurement (mg/L)
    mass_d_o_per_layer : numpy.ndarray(M, N_unique)
        Mass of dissolved oxygen per layer (tonnes)
    mass_d_o_per_period : numpy.ndarray(N_unique)
        Total mass for each time period summed across all layers (tonnes)
    inc_d_o : numpy.ndarray(M, N_unique)
        Fraction of total mass in each layer for each time period (unitless)
    cumulative_d_o : numpy.ndarray(M, N_unique)
        Cumulative percent of mass in a layer (from the bottom)

    Notes
    -----
    1)  M is the number of elevation midpoints from the bathymetric survey data
    2)  N_total is the total number of time periods represented in the datafile
    3)  N_unique is the number of unique time periods represented in the datafile
    4)  The jagged attributes are nested lists.  The first index corresponds
        to the same index as np_times_unique, the second index corresponds to
        each sampling elevation made for that time period.

    """

    def __init__(self, depth_measurements_fn):
        """Initialize attributes and read water quality information stored in a measurement csv file.

        Parameters
        ----------
        depth_measurements_fn : str
            File name with water quality measurements made at depth
        """
        # Initialize all object variables in constructor
        self.wse_interpolated = None
        self.measurement_df = None
        self.np_times = None
        self.np_times_unique = None
        self.measurements = None
        self.elevation = None
        self.temperature = None
        self.d_o = None
        self.m_temperature = None
        self.m_d_o = None
        self.mass_d_o_per_layer = None
        self.mass_d_o_per_period = None
        self.inc_d_o = None
        self.cumulative_d_o = None

        # Initialize lists that will store data in jagged arrays
        # Each index is associated with the np_times_unique index
        self.jagged_elevations = []
        self.jagged_temperature = []
        self.jagged_d_o = []

        # Read in depth measurements from a csv file
        self.depth_measurements_fn = depth_measurements_fn
        self.read_measurements(depth_measurements_fn, False)
        self.num_measurements = np.size(self.np_times)

    def read_measurements(self, depth_measurements_fn, plot_data=False):
        """Read measurements such as temperature and dissolved oxygen from cvs file.

        Parameters
        ----------
        depth_measurements_fn : str
            File name with water quality measurements made at depth
        plot_data : {True, False}
            Determines if measurements are plotted

        Notes
        -----
        1)  Measurement file header is:
                Date, Elevation (ft AMSL), Temp (oC), DO (mg/L)
        2)  The measurements file has sampling dates that may not correspond to days
            that water surface elevation was measured, so wse's may have to be interpolated.
        3)  The original depth measurements file has data up to 2016-09-26,
            whereas we only have wse measurements up to 2014-12-01.  The depth measurements file
            data was truncated wo that it excludes measurements made after 2014-12-01
        4)  Modifies the following Attributes
                measurement_df,
                np_times,
                measurements,
                elevation,
                temperature,
                d_o
        """

        # Read csv file into pandas dataframe
        self.measurement_df = pd.read_csv(depth_measurements_fn, index_col=0, parse_dates=True)

        # Sort data 1st by date ascending, then 2nd by elevation ascending
        self.measurement_df.sort_values(by=['Date', 'Elevation (ft AMSL)'], ascending=[True, True], inplace=True)

        # Extract pandas data as numpy matrices
        self.np_times = self.measurement_df.index.to_numpy(dtype=np.datetime64)
        self.measurements = self.measurement_df.to_numpy()
        self.elevation = self.measurements[:, 0]
        self.temperature = self.measurements[:, 1]
        self.d_o = self.measurements[:, 2]

        if plot_data:
            self.measurement_df.plot(marker='o', markerfacecolor='r')

    def prep_data(self, bath, wse_measured, plot_data=False):
        """Prepare measurements at depths for visualization/calculations.

        Parameters
        ----------
        bath : Bathymetry
            Bathymetry of lake to be modeled
        wse_measured : WaterSurfaceElevations
            Surveyed surface water elevations and associated lake volume

        Notes
        -----
        1)  Modifies the following Attributes:
                np_times_unique,
                wse_interpolated
        """

        # find the unique times that below water measurements were made
        self.np_times_unique = np.unique(self.np_times)
        # print(f'# of unique interpolations = {np.shape(np_times_unique)}')

        plt.show()

        # Linearly interpolate wse survey data to estimate wse when depth measurements were made
        if plot_data:
            wse_interp = InterpFunctions.interpolate_wse(
                wse_measured.np_wse_times, wse_measured.wse, self.np_times_unique, self.np_times, self.elevation, True)
        else:
            wse_interp = InterpFunctions.interpolate_wse(
                wse_measured.np_wse_times, wse_measured.wse, self.np_times_unique, self.np_times, self.elevation, False)

        plt.show()

        # Save the interpolated wse time series
        self.wse_interpolated = WaterSurfaceElevations()
        self.wse_interpolated.set_wse_and_time(self.np_times_unique, wse_interp)

        # Add wse time series to bathymetry so that it is augmented with plotting capacity
        bath.add_wse("Depth Measurements", self.wse_interpolated)

        # Group the depth measured data into jagged arrays for each unique timestamp
        self.group_depth_measurements()

        # Interpolate the grouped depth measurements into 2d numpy arrays
        self.interpolate_grouped_measurements()

        # Find the fraction of dissolved oxygen in each layer,
        # and the total DO summed across each layer for each time period.
        self.find_incremental_d_o()

    def group_depth_measurements(self):
        """Save the elevation, temperature, and DO information that is stored in numpy arrays
        into jagged data structures that can be easily interpolated and plotted.

        Notes
        -----
        1) Since the numpy arrays are sorted by time, elevation, you can use the difference in two
            time signatures to see if the date has changed for the next row of measurement data.
        2) Modifies the following Attributes:
                jagged_elevations
                jagged_temperature
                jagged_d_o
        3) The first index of each jagged array matches those of np_times_unique
        """

        # min resolution in time data (any smaller than this is considered 0)
        time_precision = 0.001

        # Convert date/time data to float for ease of calculations
        t0 = self.np_times.astype('float64')

        # Group the measurement data by finding when the time indices are different
        # By prepending a 0 and shifting these indices, you can make start/stop
        #   locations to naturally group data that have the same timestamp

        # find time differences between two measurements
        t1 = np.diff(t0)
        # find where the diff's are non-zero
        [t2] = np.where(t1 > 0)
        # shift the index by one since a diff vector is one shorter than its orginal
        t2 += 1

        # Find the start and stop indices associated with each time period

        # start will always include the first index
        starts = np.concatenate(([0], t2))

        # end will always include last the last member of time series
        ends = np.concatenate((t2, [np.size(t0)]))

        # number of depths measured at each time period
        num_at_depth = ends - starts

        # print(t0[0:20])
        # print(t1[0:20])
        # print(t2[0:20])
        # print(starts, ends, num_at_depth)

        # Group measurement data into jagged arrays using start and end index information
        # Each list will be the same length as np_times_unique
        for i, j in zip(starts, ends):
            self.jagged_elevations.append(self.elevation[i:j])
            self.jagged_temperature.append(self.temperature[i:j])
            self.jagged_d_o.append(self.d_o[i:j])

        # print(self.np_times[0])
        # print(my_elevations[0])
        # print(my_temperature[0])
        # print(my_d_o[0])
        #
        # print(self.np_times[-1])
        # print(my_elevations[-1])
        # print(my_temperature[-1])
        # print(my_d_o[-1])

    def interpolate_grouped_measurements(self):
        """Interpolate the depth measurements into a 2d numpy matrices
        where each row is the elevation midpoint for that time period
        and each column is associated with a unique measurement period.

        Notes
        -----

        """

        # Find the midpoints and shape of the matrices to store interpolations
        midpoints = self.wse_interpolated.m_mid_elevation
        num_row, num_col = np.shape(midpoints)
        # print(num_row, num_col)

        # Initialize interpolated matrices to negative 1 to make error detection easier
        self.m_temperature = np.ones((num_row, num_col)) * -1
        self.m_d_o = np.ones((num_row, num_col)) * -1

        # Loop through each unique time period
        for j in range(num_col):

            # The number of data points for each time period j can vary
            # That is why jagged arrays are used
            z = self.jagged_elevations[j]
            temp = self.jagged_temperature[j]
            d_o = self.jagged_d_o[j]

            # Find topmost and bottom most temperature and dissolved oxygen measurements
            # anything above the top layer measurement will set to the top layer measurement
            # anything below the bottom layer measurement will be set to the bottom layer measurement
            bottom_temp = temp[0]
            top_temp = temp[-1]
            bottom_d_o = d_o[0]
            top_d_o = d_o[-1]
            # print(bottom_temp, top_temp)

            # Create interpolation functions based on measured data
            f_temp = interpolate.interp1d(z, temp, fill_value=(bottom_temp, top_temp), bounds_error=False)
            f_d_o = interpolate.interp1d(z, d_o, fill_value=(bottom_d_o, top_d_o), bounds_error=False)

            # Use the interpolation functions to estimate temperature and DO at each midpoint
            new_temp = f_temp(midpoints[:, j])
            new_d_o = f_d_o(midpoints[:, j])

            # print(self.m_temperature[:, j])
            # print(new_temp)
            # print(self.m_d_o[:, j])
            # print(new_d_o)

            # Save the interpolated values into attributes
            self.m_temperature[:, j] = new_temp
            self.m_d_o[:, j] = new_d_o

    def find_incremental_d_o(self):
        """Find the fraction of dissolved oxygen in each layer, and the total DO summed across
        each layer for each time period.

        Notes
        -----
        """

        # Find the midpoints and shape of the matrices to store interpolations
        midpoints = self.wse_interpolated.m_mid_elevation  #(feet)
        num_row, num_col = np.shape(midpoints)

        # Shortcuts to volume and dissolved oxygen for each layer
        volume_af = self.wse_interpolated.m_vol_inc_af  # (acre-feet)
        d_o_mg = self.m_d_o  # mg/liter (for each layer)

        # Convert units
        volume_l = volume_af * LakeMeasurements.L_PER_ACRE_FT  # (liters)
        unit_conversion = LakeMeasurements.MG_PER_KG * LakeMeasurements.KG_PER_TONNE
        d_o_tonnes = d_o_mg / unit_conversion  # (tonnes/l)

        # Determine mass of DO per layer (tonnes) by
        #  multiplying water volume (liters) by concentration of DO (tonnes/liters) in layer
        self.mass_d_o_per_layer = volume_l * d_o_tonnes   # tonnes (for each layer)

        # Sum the DO (tonnes) across each layer for each time period
        self.mass_d_o_per_period = np.sum(self.mass_d_o_per_layer, axis=0)  # (tonnes)

        # Expand the sum vector to a matrix to make subsequent calculations easier
        mass_d_o_per_period_tiled = np.tile(self.mass_d_o_per_period, (num_row, 1))  # (tonnes)

        # Find the fraction of total lake DO in each layer (unitless)
        self.inc_d_o = self.mass_d_o_per_layer / mass_d_o_per_period_tiled

        # Calculate the cumulative DO from the bottom of the lake to the top
        self.cumulative_d_o = np.cumsum(self.inc_d_o, axis=0)

        # print(np.shape(self.m_d_o), np.shape(self.mass_d_o_per_period), np.shape(mass_d_o_per_period_tiled))


    def plot_interpolated_depth(self, i=0, fig=None, ax0=None, ax1=None):
        """Plot temperature dissolved oxygen profile from interpolated measurements.

        Parameters
        ----------
        i : integer
            Index of time to plot
        fig : figure.Figure
            Figure containing axis ax0 and ax1
        ax0 : axes.Axes
            Axis used to plot temperature data
        ax1 : axes.Axes
            Axis used to plot dissolved oxygen data

        Notes
        -----
        """

        if fig is None:
            fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

        # Shortcuts to sample times, layer midpoints
        mytime = np.datetime_as_string(self.np_times_unique[i], unit='D')
        midpoints = self.wse_interpolated.m_mid_elevation

        # Shortcuts to measured data
        measure_z = self.jagged_elevations[i]
        measure_t = self.jagged_temperature[i]
        measure_d = self.jagged_d_o[i]

        # Shortcuts to interpolated data
        interp_z = midpoints[:, i]
        interp_t = self.m_temperature[:, i]
        interp_d = self.m_d_o[:, i]

        # Plot measured and interpolated data
        ax0.plot(interp_t, interp_z, 'bx-', label="Interpolated")
        ax0.plot(measure_t, measure_z, 'ro', label="Measured")
        ax1.plot(interp_d, interp_z, 'bx-')
        ax1.plot(measure_d, measure_z, 'ro')

        # label figure
        ax0.set_ylabel('Elevation (ft)')
        ax0.set_xlabel('Temperature (C)')
        ax1.set_xlabel('Dissolved Oxygen (mg/L)')
        fig.suptitle(f'Measurements on {mytime}')
        fig.legend()

    def plot_profile_slider(self, plot_type='line'):
        """Visualize the depth profiles with a slider that allows you to select which day you want to visualize.

        Plot can visualize profiles as either lines or colored polygons based on the plot_type.

        Parameters
        ----------
        plot_type : str {'line', 'polygon'}

        Returns
        -------
        fig : figure.Figure
            Figure containing axis lake_ax and slider_ax
        ax_temp : axes.Axes
            Axis to plot temperature data
        ax_d_o : axes.Axes
            Axis to plot dissolved oxygen data
        slider_ax : axes.Axes
            Axis for slider to select measurement day

        Notes
        -----
        """
        # todo - compare results to spreadsheet figures and calcs

        # Find the midpoints and shape of the matrices to store interpolations
        midpoints = self.wse_interpolated.m_mid_elevation  # (feet)
        num_row, num_col = np.shape(midpoints)

        # % Contribution of daily DO per layer
        # Convert fractions to percents by multiplying by 100
        lake_do_per_layer = self.inc_d_o * 100
        lake_cumulative_d_o = self.cumulative_d_o * 100

        # Find min and max of temperature and DO
        min_temp = np.min(self.m_temperature.flatten())
        max_temp = np.max(self.m_temperature.flatten())
        min_d_o = np.min(self.m_d_o.flatten())
        max_d_o = np.max(self.m_d_o.flatten())
        #  print(min_temp, max_temp, min_d_o, max_d_o)

        # Plot limits are now hard coded to Loch Lomond, but can be generalized in future applications
        # limits are currently hard codded, consider making them a generalizable function
        # temp_lim = (min_temp*0.9, max_temp*1.1)
        # do_lim = (min_d_o*0.9, max_d_o*1.1)
        temp_lim = (5, 25)
        do_lim = (0, 15)
        ylim = (420, 580)
        poly_lim = (-5, 175)
        ylim = (420, 580)

        # Create a figure with an axis to show lake data and an axis for the slider
        fig, (ax_temp, ax_d_o, ax_slider, ax_cum_d_o) = plt.subplots(4, 1, figsize=(8, 3))
        ax_temp.set_position([0.1, 0.2, .25, 0.65])
        ax_d_o.set_position([0.4, 0.2, .25, 0.65])
        ax_cum_d_o.set_position([0.7, 0.2, .25, 0.65])

        ax_slider.set_position([0.1, 0.05, 0.8, 0.05])

        # Create the slider and the function to be executed when slider value changes

        # Determine the range of index values considered for the slider
        day_min = 0  # the minimal value of the parameter a
        day_max = num_col - 1  # the maximal value of the parameter a
        day_init = 0  # the value of the parameter a to be used initially, when the graph is created
        day_step = 1  # incremental value for the slider

        # Create the slider
        day_slider = Slider(ax_slider,  # the axes object containing the slider
                            'day',  # the name of the slider parameter
                            day_min,  # minimal value of the parameter
                            day_max,  # maximal value of the parameter
                            valinit=day_init,  # initial value of the parameter
                            valstep=day_step,  # step increments for the slider
                            valfmt='%1.0f'
                            )

        # Define a function that will be executed each time the value
        # indicated by the slider changes. The variable of this function will
        # be assigned the value of the slider.
        def update(d):
            index = int(d)  # find day of interest
            mytime = np.datetime_as_string(self.wse_interpolated.np_wse_times[index], unit='D')

            # print(index)
            ax_temp.clear()
            ax_d_o.clear()
            ax_cum_d_o.clear()

            # Plot cumulative DO mass from bottom of lake to midpoint
            ax_cum_d_o.plot(lake_cumulative_d_o[:, index], midpoints[:, index], 'g+-')
            ax_cum_d_o.set_xlabel('% Below Midpoint')
            ax_cum_d_o.set_title('DO Cumulative Mass')

            if plot_type == 'line':
                self.plot_interpolated_depth(index, fig, ax_temp, ax_d_o)
                ax_temp.set_xlim(temp_lim)
                ax_d_o.set_xlim(do_lim)

            elif plot_type == 'polygon':
                # plot polygons associated with selected day
                Bathymetry.plot_polygons(self.wse_interpolated.patches[index],
                                         ax_temp,
                                         values=self.m_temperature[:, index],
                                         max_value=max_temp)
                Bathymetry.plot_polygons(self.wse_interpolated.patches[index],
                                         ax_d_o,
                                         values=self.m_d_o[:, index],
                                         max_value=max_d_o)
                ax_temp.set_xlim(poly_lim)
                ax_d_o.set_xlim(poly_lim)
                ax_temp.set_ylabel('Elevation (ft)')
                ax_temp.set_xlabel('Normalized Cross-Sectional Area')
                ax_d_o.set_xlabel('Normalized Cross-Sectional Area')
                ax_temp.set_title('Temperature')
                ax_d_o.set_title('Dissolved Oxygen')
                fig.suptitle(f'Measurements on {mytime}')

            else:
                print(f"Unknown plot type requested: {plot_type}")
                return

            ax_temp.set_ylim(ylim)
            ax_d_o.set_ylim(ylim)
            ax_cum_d_o.set_ylim(ylim)
            fig.canvas.draw_idle()  # redraw the plot

        # Specify that the slider needs to execute the update() function when its value changes
        day_slider.on_changed(update)

        # Initialize the plot to visualize the 1st index of the slider
        update(0)

        # Figures with sliders must be show before other plots are visualized for the slider to work properly
        plt.show()

        return fig, (ax_temp, ax_d_o, ax_slider)


class InterpFunctions:
    """Class to store interpolation functions and to test various ways to interpolate lake data.

    Notes
    -----
    1) This class is in initial development, and is clunky, but it can be refined,
        and cleaned up later.
    """

    # todo - compare interpolation routines to spreadsheet analysis

    @staticmethod
    def interpolate_wse(wse_times, wse, new_times,
                        depth_meas_times=None, depth_meas_elevations=None, plot_data=False):
        """Interpolate an estimate of wse on depth measurement dates
        based on linear interpolation of wse measurements.

        Parameters
        ----------
        wse_times : numpy.ndarray() of type np.datetime64
            Times associated with wse survey measurements.
        wse : numpy.ndarray
            Water surface elevations associated with surveyed wse_times (feet)
        new_times : numpy.ndarray() of type np.datetime64
            Times that require interpolated estimates of wse values.
        depth_meas_times : numpy.ndarray() of type np.datetime64, optional
            Sampling times associated with meas_elevations.
        depth_meas_elevations : numpy.ndarray, optional
            Elevations associated with underwater measurements.
        plot_data : {True, False}
            True if you want to plot data

        Returns
        -------
        wse_interpolated :
            estimated wse values at depth measurement times

        Notes
        -----
        1) depth_meas_elevations are just the elevations, not the measurements made at that elevation.
            Plotting these values gives a quick visualization of what elevations data were sampled.
        """

        # print(f'type(self.np_wse_times) = {type(wse_times)}')
        # print(f'shape self.np_wse_times = {np.shape(wse_times)}')
        # print(f'min self.np_wse_times = {np.min(wse_times)}')
        # print(f'max self.np_wse_times = {np.max(wse_times)}')

        # print(f'type(self.wse2) = {type(wse)}')
        # print(f'shape self.wse2 = {np.shape(wse)}')
        # print(f'min self.np_ws2 = {np.min(wse)}')
        # print(f'max self.np_ws2 = {np.max(wse)}')

        # print(f'type(self.np_times_unique) = {type(new_times)}')
        # print(f'shape self.np_times_unique = {np.shape(new_times)}')
        # print(f'min self.np_times_unique = {np.min(new_times)}')
        # print(f'max self.np_times_unique = {np.max(new_times)}')

        # Create linear interpolation function and estimate wse at new times
        # Note, times have to be cast to 'float64' for interp1d to work properly
        f = interpolate.interp1d(wse_times.astype('float64'), wse)
        wse_interpolated = f(new_times.astype('float64'))

        if plot_data:
            if (depth_meas_times is None) or (depth_meas_elevations is None):
                InterpFunctions.plot_sampling_info(wse_times, wse, new_times, wse_interpolated)
            else:
                InterpFunctions.plot_sampling_info(wse_times, wse, new_times, wse_interpolated,
                                                   depth_meas_times, depth_meas_elevations)

        return wse_interpolated

    @staticmethod
    def plot_sampling_info(wse_times, wse, interp_times, interp_wse,
                           depth_meas_times=None, depth_meas_elevations=None,
                           plot_axis=None):
        """Visualize surveyed and interpolated water surface elevation along with the option
        to show location/time of depth measurements.

        Parameters
        ----------
        wse_times : numpy.ndarray() of type np.datetime64
            Times associated with wse survey measurements.
        wse : numpy.ndarray
            Water surface elevations associated with surveyed wse_times (feet)
        interp_times : numpy.ndarray() of type np.datetime64
            Times that wse are interpolated.
        interp_wse : numpy.ndarray
            Interpolated wse associated with interp_times (feet)
        depth_meas_times : numpy.ndarray() of type np.datetime64, optional
            Sampling times associated with meas_elevations.
        depth_meas_elevations : numpy.ndarray, optional
            Elevations associated with underwater measurements.
        plot_axis : axes.Axes

        Returns
        -------
        ax : axes.Axes
            Axis used to visualize measured and interpolated values

        Notes
        -----
        1) depth_meas_elevations are just the elevations, not the measurements made at that elevation.
            Plotting these values gives a quick visualization of what elevations data were sampled.
"""

        if plot_axis is None:
            fig, plot_axis = plt.subplots(1)

        title = 'Measured/Interpolated WSE'

        plot_axis.plot(wse_times, wse, 'bo-', label='Surveyed WSE')
        plot_axis.plot(interp_times, interp_wse, 'r+', label='Interpolated WSE')

        if (depth_meas_times is not None) and (depth_meas_elevations is not None):
            title += " & Depth Measurement Locations"
            plot_axis.plot(depth_meas_times, depth_meas_elevations, 'gx', markersize=1, label='Depth Measurements')

        plot_axis.set_ylabel('Water Surface Elevation (feet)')
        plot_axis.set_xlabel('WSE and Depth Measurement Time')

        plot_axis.set_title(title)
        plot_axis.legend()

        return fig, plot_axis


if __name__ == '__main__':
    """
    Main program to visualize Lake Data.
    Lake data are read from csv file, or optionally from pickle files to speed development.
    Primary calcs and visualizations are performed in mylake.driver().
    """
    my_timer = MyTimer("Main Program")
    # Option to read data or save data to/from pickle files to speed up development
    read_from_pickle = False
    save_to_pickle = True

    # File to store/load lake information
    fn_pickle = 'mylake.pickle'

    # Read data from pickle or csv files
    if read_from_pickle:
        with open(fn_pickle, 'rb') as f:
            mylake = pickle.load(f)
    else:
        mylake = LakeMeasurements()
        mylake.read_csv_files()

    if save_to_pickle:
        with open(fn_pickle, 'wb') as f:
            pickle.dump(mylake, f)

    mylake.driver()
    plt.show()

    # Optional diagnostic routines to explore new functionality
    # mylake._test_diagnostics()

    my_timer.toc()
