# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# py-sphviewer2: High-performance SPH visualization
#
# Copyright (c) 2024 Alejandro Benitez-Llambay
# Distributed under the terms of the MIT License.
# -----------------------------------------------------------------------------
"""
Python interphase for py-sphviewer2
"""

import math
import numpy as np
from . import core
import h5py
from scipy.spatial.transform import Rotation as R

# Expose the SPHData struct so users can access it easily
SPHData = core.SPHData

class Particles:
    """
    A data container that prepares and standardizes SPH particle arrays for the C++ backend.

    This class ensures that all input arrays (coordinates, smoothing lengths, and masses)
    are cast to 64-bit floats and structured in contiguous C-style memory blocks. This is
    a critical step that allows the C++ extension to access NumPy memory directly via
    zero-copy buffer protocols, avoiding expensive RAM duplication.

    Attributes
    ----------
    x, y, z : ndarray
        1D contiguous arrays of particle coordinates (float64).
    h : ndarray
        1D contiguous array of particle smoothing lengths (float64).
    m : ndarray
        1D contiguous array of particle masses or weights (float64).
    """

    def __init__(self, x, y, z, h, m):
        """
        Initialize the Particles object.

        Parameters
        ----------
        x : array_like
            1D array of particle X coordinates.
        y : array_like
            1D array of particle Y coordinates.
        z : array_like
            1D array of particle Z coordinates.
        h : array_like
            1D array of particle smoothing lengths.
        m : array_like
            1D array of particle masses (or any scalar field you wish to project).

        Notes
        -----
        All input arrays are automatically converted to `np.float64` and forced into
        a contiguous memory layout (`np.ascontiguousarray`). This safely handles 
        sliced N-dimensional data (e.g., passing `pos[:, 0]` from an HDF5 file) 
        without causing segmentation faults when read by C++.
        """

        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.z = np.ascontiguousarray(z, dtype=np.float64)
        self.h = np.ascontiguousarray(h, dtype=np.float64)
        self.m = np.ascontiguousarray(m, dtype=np.float64)

    def _pack(self):
        """
        Pack the sanitized arrays into the C++ SPHData struct.

        Returns
        -------
        core.SPHData
            The PyBind11-bound C++ struct containing the array memory pointers,
            ready to be passed directly into the Projector.

        Notes
        -----
        This is an internal method called automatically by the `Projector` class 
        during the rendering phase. Users typically do not need to call this manually.
        """

        data = core.SPHData()
        data.x, data.y, data.z, data.h, data.m = self.x, self.y, self.z, self.h, self.m
        return data

class Camera:
    """
    Defines the field-of-view, 3D orientation, and resolution parameters for the renderer.

    The Camera class is responsible for computing the 3D rotation matrix and managing 
    the spatial boundaries of the projection. It dictates where the renderer looks, 
    how closely it zooms, and the quality/resolution of the underlying grids.
    """
    def __init__(self, Lbox, extent=None, xc=None, yc=None, zc=None,
                 azimuth=0, elevation=0, roll=0,
                 periodic=True, target_cells_per_h=4, r_max=9, r_min=None):
        """
        Initialize the Camera object.

        Parameters
        ----------
        Lbox : float
            The physical size of the simulation box. Required for periodic wrapping.
        extent : float, optional
            The physical width/height of the camera's field of view. If None, 
            defaults to the full `Lbox`. Decreasing this effectively "zooms in".
        xc, yc, zc : float, optional
            The 3D coordinates of the target center (the pivot point for rotation).
            Defaults to the center of the box (`Lbox / 2.0`).
        azimuth : float, optional
            Intrinsic rotation around the Z-axis in degrees. Default is 0.
        elevation : float, optional
            Intrinsic rotation around the X-axis in degrees. Default is 0.
        roll : float, optional
            Intrinsic rotation around the Y-axis in degrees. Default is 0.
        periodic : bool, optional
            If True, particles wrap around the boundaries defined by `Lbox`. Default is True.
        target_cells_per_h : int, optional
            The quality factor of the projection ($N_h$). Represents the minimum number 
            of grid cells that sample a particle's smoothing radius. Default is 4.
        r_max : int, optional
            The maximum resolution level. The final image will be 2^r_max x 2^r_max pixels.
        r_min : int, optional
            The minimum resolution level. If None, it is automatically calculated based 
            on `target_cells_per_h` to ensure optimal memory usage.
        """
        self.Lbox = Lbox
        # Extent defaults to the full box if not zooming
        self.extent = extent if extent is not None else Lbox
        
        self.xc = xc if xc is not None else Lbox / 2.0
        self.yc = yc if yc is not None else Lbox / 2.0
        self.zc = zc if zc is not None else Lbox / 2.0        
        self.periodic = periodic
        
        self.target_cells_per_h = target_cells_per_h
        self.r_max = r_max
        self.r_min = r_min if r_min is not None else int(math.log2(target_cells_per_h)) + 1
    
        # Generate the rotation matrix from Euler angles (degrees)
        rot = R.from_euler('XYZ', [elevation, azimuth, roll], degrees=True)
        self.rot_matrix = rot.as_matrix().flatten().astype(np.float64)

    def get_extent(self):
        """
        Calculates the 2D bounding box of the camera's view for plotting.

        Returns
        -------
        list
            A list containing [xmin, xmax, ymin, ymax] in physical units.
            This is formatted perfectly for the `extent` argument in `matplotlib.pyplot.imshow`.
        """        
        half_ext = self.extent / 2.0
        return [self.xc - half_ext, self.xc + half_ext, 
                self.yc - half_ext, self.yc + half_ext]

class Projector:
    """
    The main rendering engine that interfaces with the C++ backend.

    The Projector manages the hierarchy of sparse nested grids. It handles the 
    heavy computational lifting of projecting particles into these grids, collapsing 
    them into a final 2D image, and saving/loading the grid states to disk for deferred rendering.
    """
    def __init__(self, camera):
        """
        Initialize the Projector with a specific Camera configuration.

        Parameters
        ----------
        camera : sphviewer2.Camera
            The camera object defining the view, rotation, and resolution.
        """
        self.camera = camera
        # Initialize grids using the extent, NOT Lbox!
        self._grids = core.NestedGrids(
            camera.extent, 
            camera.target_cells_per_h, 
            camera.r_min, 
            camera.r_max
        )

    def project(self, particles, num_threads=4):
        """
        Projects the particles into the nested grid hierarchy.

        This is the core computational step. It distributes the particles across 
        threads and projects them onto their native resolution levels based on 
        their smoothing lengths, applying 3D rotations on-the-fly.

        Parameters
        ----------
        particles : sphviewer2.Particles
            The data container holding the coordinates, smoothing lengths, and masses.
        num_threads : int, optional
            The number of CPU threads to use for parallel projection. Default is 4.
        """
        self._grids.project(
        particles._pack(), 
        num_threads, 
        self.camera.extent,
        self.camera.Lbox,
        self.camera.xc,
        self.camera.yc,
        self.camera.zc,
        self.camera.rot_matrix,
        self.camera.periodic
        )
    
    def save_baked_grids(self, filename, float32=True):
        """
        Saves the populated nested grids and camera metadata to an HDF5 file.

        This enables "Deferred Rendering". By baking the grids to disk, you can 
        bypass the heavy `project()` step in future sessions and instantly explore 
        the resolution scales or collapse the final image.

        Parameters
        ----------
        filename : str
            The output path for the HDF5 file (e.g., 'simulation_baked.hdf5').
        float32 : bool, optional
            If True, casts the grid data from float64 to float32 before saving 
            to halve the file size. Default is True.
        """
        with h5py.File(filename, 'w') as f:
            # Save Metadata to reconstruct the Projector state
            m = f.create_group("metadata")
            m.attrs['Lbox'] = self.camera.Lbox
            m.attrs['extent'] = self.camera.extent
            m.attrs['xc'] = self.camera.xc
            m.attrs['yc'] = self.camera.yc
            m.attrs['zc'] = self.camera.zc
            m.attrs['azimuth'] = self.camera.azimuth
            m.attrs['elevation'] = self.camera.elevation
            m.attrs['roll'] = self.camera.roll
            m.attrs['periodic'] = self.camera.periodic
            m.attrs['r_min'] = self.camera.r_min
            m.attrs['r_max'] = self.camera.r_max
            m.attrs['target_cells_per_h'] = self.camera.target_cells_per_h
            
            # Save each level with GZIP compression
            g = f.create_group("grids")
            for r in range(self.camera.r_min, self.camera.r_max + 1):
                data = self._grids.get_level_data(r)
                
                # Reduce space if needed
                if float32:
                    data = data.astype(np.float32)

                # Compression is key for sparse SMESHL grids!
                g.create_dataset(f"level_{r}", data=data, 
                                 compression="gzip", 
                                 compression_opts=4,
                                 shuffle=True)

    @classmethod
    def load_baked_grids(cls, filename):
        """
        Factory method to reconstruct a Projector from a baked HDF5 file.

        Parameters
        ----------
        filename : str
            The path to the previously baked HDF5 file.

        Returns
        -------
        sphviewer2.Projector
            A fully initialized Projector object with pre-populated grids, 
            ready to be collapsed or explored.
        """
        with h5py.File(filename, 'r') as f:
            m = f['metadata'].attrs
            
            # Reconstruct Camera and Projector
            cam = Camera(Lbox=m['Lbox'], extent=m['extent'], 
                         xc=m['xc'], yc=m['yc'], zc=m['zc'],
                         azimuth=m['azimuth'], elevation=m['elevation'], roll=m['roll'],
                         periodic=m['periodic'],
                         target_cells_per_h=m['target_cells_per_h'], 
                         r_max=m['r_max'], r_min=m['r_min'])
            proj = cls(cam)
            
            # Inject data back into C++
            for r in range(m['r_min'], m['r_max'] + 1):
                proj._grids.set_level_data(r, f[f"grids/level_{r}"][()])
                
            return proj

    def collapse(self):
        """
        Sums up the grid hierarchy to produce the final high-resolution image.

        This method bilinearly interpolates all lower-resolution grids upward 
        and adds them to the highest-resolution grid (`r_max`). This guarantees 
        mass conservation across all spatial scales.

        Returns
        -------
        ndarray
            A 2D NumPy array (size 2^r_max x 2^r_max) representing the final projected field.
        """
        return self._grids.collapse()    

    def get_level(self, level):
        """
        Extracts the raw 2D grid data for a specific resolution level.

        Parameters
        ----------
        level : int
            The target resolution level $R$. Must be between `r_min` and `r_max`.

        Returns
        -------
        ndarray
            A 2D NumPy array of size 2^R x 2^R containing the data projected 
            exclusively at this spatial scale.
        """
        return self._grids.get_level_data(level)   
         

def render(x, y, z, h, m, Lbox=1.0, extent=None, 
           xc=None, yc=None, zc=None,
           azimuth=0, elevation=0, roll=0,
           periodic=True, r_max=9, target_cells_per_h=4, num_threads=4):
    
    """
    A high-level wrapper to directly render SPH particles into a 2D density map.

    This function automatically handles the creation of the `Particles`, `Camera`, 
    and `Projector` objects. It projects the particles into a multi-scale 
    nested grid hierarchy and immediately collapses them into a final 
    high-resolution 2D image.

    Parameters
    ----------
    x, y, z : array_like
        1D arrays of particle coordinates.
    h : array_like
        1D array of particle smoothing lengths.
    m : array_like
        1D array of particle masses or weights.
    Lbox : float, optional
        The size of the periodic simulation box. Default is 1.0.
    extent : float, optional
        The physical width/height of the camera's field of view. 
        If None, defaults to the full `Lbox`.
    xc, yc, zc : float, optional
        The 3D coordinates of the camera target center. Defaults to `Lbox / 2.0`.
    azimuth, elevation, roll : float, optional
        Intrinsic 3D Euler angles (in degrees) for the camera rotation. Default is 0.
    periodic : bool, optional
        If True, wraps particles around the boundaries defined by `Lbox`. Default is True.
    r_max : int, optional
        The maximum resolution level. Output image will be 2^r_max x 2^r_max pixels. Default is 9.
    target_cells_per_h : int, optional
        The projection quality factor ($N_h$). Default is 4.
    num_threads : int, optional
        Number of CPU threads to use for parallel projection. Default is 4.

    Returns
    -------
    image : ndarray
        A 2D NumPy array representing the projected density field.
    extent : list
        A list `[xmin, xmax, ymin, ymax]` indicating the physical bounds of the image. 
        Can be passed directly to `matplotlib.pyplot.imshow(..., extent=extent)`.
    """    

    p = Particles(x, y, z, h, m)
    c = Camera(
        Lbox=Lbox, extent=extent, 
        xc=xc, yc=yc, zc=zc,
        azimuth=azimuth, elevation=elevation, roll=roll,
        periodic=periodic, r_max=r_max, target_cells_per_h=target_cells_per_h
        )
    
    proj = Projector(c)
    
    img = proj.project(p, num_threads=num_threads)
    image = proj.collapse()

    return image, c.get_extent()

def estimate_h(x, y, z=None, Lbox=1.0, k=32, num_threads=4):
    """
    Estimates the exact SPH smoothing length (h) for each particle.

    This function utilizes a high-performance, multi-threaded C++ backend to 
    perform an exact k-nearest neighbor (KNN) search across periodic boundaries,
    returning the radius required to enclose exactly `k` neighbors.

    Parameters
    ----------
    x, y : array_like
        1D arrays of particle X and Y coordinates.
    z : array_like, optional
        1D array of particle Z coordinates. If None, the search assumes a 
        purely 2D plane (z=0 for all particles).
    Lbox : float, optional
        The size of the simulation box. Required to correctly compute neighbor 
        distances across periodic boundaries. Default is 1.0.
    k : int, optional
        The target number of neighbors to enclose within the smoothing radius. 
        Higher values produce smoother fields; lower values preserve sharp features. 
        Default is 32.
    num_threads : int, optional
        Number of CPU threads to use for the parallel neighbor search. Default is 4.

    Returns
    -------
    h : ndarray
        A 1D NumPy array of `float64` containing the calculated smoothing 
        lengths for every input particle.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    
    if z is None:
        z = np.zeros_like(x)
    else:
        z = np.ascontiguousarray(z, dtype=np.float64)
        
    return core.estimate_smoothing_length(x, y, z, Lbox, k, num_threads)