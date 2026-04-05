# py-sphviewer2

**py-sphviewer2** is a high-performance, multithreaded C++ library with a Python interface designed for the rapid visualization of Smoothed Particle Hydrodynamics (SPH) simulations. It implements a novel algorithm described in Benitez-Llambay (2025), providing an efficient approach to rendering any field traced by particles using the Smoothed Particle Hydrodynamics (SPH) framework, with a time complexity that is independent of the final image resolution. 

## Why py-sphviewer2?

Traditional SPH visualization tools typically use a "scatter" approach, where every particle's kernel is deposited onto a grid. As the desired image resolution increases, the number of pixels covered by a single particle kernel grows quadratically, leading to a massive computational bottleneck, largely dominated by low-density particles.

**py-sphviewer2** bypasses this entirely. By using a hierarchy of nested grids, the algorithm ensures that the number of operations per particle remains constant, regardless of whether you are rendering a 512x512 image or a 16384x16384 poster.

### Key Features:
* **Constant Numerical Complexity:** Projection time is decoupled from image resolution.
* **Zero-Copy Memory Access:** Utilizes NumPy buffer protocols for direct C++ access to Python memory.
* **Mass Conservation:** Guaranteed mass recovery via normalized kernels and bilinear grid collapsing.
* **Multithreaded Backend:** Native C++ implementation using thread-local grids for lock-free parallel projection. No more issues with OpenMP on Mac.
* **Advanced Optics:** Built-in support for periodic boundary conditions, zooming, and off-center cropping.
* **On-the-fly 3D Rotations:** Perform instantaneous 3D camera rotations with zero memory duplication using natively compiled C++ matrix transformations.
* **Fast Smoothing Lengths:** Includes a multi-threaded exact k-nearest neighbor search to compute smoothing lengths directly from raw coordinates. No more external packages needed!
* **Deferred Rendering:** Compress and save the projected multi-scale grid hierarchy to tiny HDF5 files for instant, CPU-free exploration later.

---
## The Key Algorithm:

The core of the library is the **Nested Grid** approach. Instead of projecting all particles onto a single high-resolution grid, particles are sorted into different "Levels" ($R$) based on their smoothing length ($h$).

1.  **Level Assignment:** A particle is assigned to a grid level $R$ where its smoothing length is sampled by a fixed number of cells ($N_h$, usually 4 or 8 per dimension is adequate).
2.  **Parallel Projection:** Each particle is projected onto its "native" grid. Large particles go to coarse grids; small particles go to fine grids. Over the years, we have learned that the parallel projection is the most used projection by the community. We have dropped perspective projection for this release.
3.  **Bilinear Collapse:** All grids are interpolated and summed upward into the highest resolution level. Because each level has exactly $2^R$ cells, this "collapse" naturally preserves mass.

---
## Reference
This implementation is based on the algorithm described in:
> *A. Benítez-Llambay (2025), "Efficient Computation of Smoothed Particle Hydrodynamic Properties on Regular Grids" (https://iopscience.iop.org/article/10.3847/2515-5172/addab2)*. **Please cite this reference if you use the code.**

---

## Installation

The easiest way to install **py-sphviewer2** is via `pip`. We provide pre-compiled binaries (wheels) for **Linux, Windows, and macOS** (both Intel and Apple Silicon). This means you do not need a C++ compiler to install the library!

```bash
pip install py-sphviewer2
```

## Building From Source

If you want to modify the code or compile it yourself, ensure you have a C++14 compatible compiler (like `clang++` or `g++`).

```bash
git clone https://github.com/yourusername/py-sphviewer2.git
cd py-sphviewer2
pip install -e .
```
---

## Performance & Parallelism

**py-sphviewer2** is built for high-performance computing (HPC) environments. The `num_threads` parameter controls the OpenMP-style parallelization of the projection phase. Some of the features achieved with this architecture are:

* **Thread-Local Accumulation:** To avoid "atomic" bottlenecks and lock contention, each thread maintains its own private stack of nested grids. 
* **Scaling:** Because threads work independently on their own memory space, the algorithm scales nearly linearly with the number of CPU cores, especially for large datasets ($N > 10^6$).
* **Memory Overhead:** To achieve lock-free parallelism, each thread $T$ (where $T$ = `num_threads`) maintains its own private grid hierarchy. The peak memory usage during the projection phase scales as:

$$\approx T \times (2^{r_{max}} \times 2^{r_{max}}) \times 8 \text{ bytes}$$

For example, rendering at $`r_max=11` with `num_threads=16` requires approximately 512 MB of temporary buffer space. Note that this memory is automatically freed once the collapse phase is complete.

---
## Minimal Example

```python
import sphviewer2
import matplotlib.pyplot as plt

# x, y, h, m are your NumPy arrays
image = sphviewer2.render(x, y, z, h, m, Lbox=1.0, r_max=10)

plt.imshow(image, cmap='magma', origin='lower')
plt.show()
```

where r_max controls the output image resolution, which is a power of 2 (2^r_max x 2^r_max pixels). Looking for more? Check out the Jupyter Notebook files in the `docs` folder.

---

## 3D Camera Rotations & Zooming

`py-sphviewer2` handles 3D transformations completely internally. It applies an intrinsic (drone-style) rotation matrix inside the C++ projection loop, meaning it costs no extra RAM and computes very efficiently.

```python
image, extent = sphviewer2.render(
    x, y, z, h, m, 
    Lbox=100.0, 
    extent=25.0,            # Zoom in to a 25.0 unit field-of-view
    xc=50.0, yc=50.0, zc=50.0, # Center the camera on a specific halo
    azimuth=45.0,           # Spin 45 degrees around the target
    elevation=60.0,         # Tilt down by 60 degrees
    r_max=10
)
```

## Fast Smoothing Length Estimation

If your dataset does not include pre-calculated smoothing lengths ($h$), `py-sphviewer2` includes an efficient, multi-threaded C++ backend to estimate them using a perfect k-nearest neighbor search across periodic boundaries.

```python
# Calculate exact smoothing lengths targeting 64 neighbors
h = sphviewer2.estimate_h(x, y, z, Lbox=100.0, k=64, num_threads=8)
```

## "Baking" and Deferred Rendering

When working with massive simulations, calculating $h$ and projecting the particles takes significant CPU time and RAM. Baking allows you to run the heavy projection step once on a supercomputer, save the sparse nested grid hierarchy to a tiny HDF5 file, and instantly explore it later on a laptop.

```python
# 1. Project and Bake (on an HPC node)
proj.project(particles, num_threads=16)
proj.save_baked_grids("my_simulation_baked.hdf5")

# 2. Instantly Load and Explore (on your laptop)
proj_loaded = sphviewer2.Projector.load_baked_grids("my_simulation_baked.hdf5")

# Extract individual resolution levels or collapse the final image instantly
diffuse_gas = proj_loaded.get_level(6)
final_image = proj_loaded.collapse()
```

## The More Modular API
For complex tasks (like animations), use the object-oriented interface to avoid re-packing data:


```python
particles = sphviewer2.Particles(x, y, zm h, m)
camera = sphviewer2.Camera(
    Lbox=1.0, 
    extent=0.1, 
    xc=0.5, yc=0.5, zc=0.5,
    azimuth=30, elevation=45
)

projector = sphviewer2.Projector(camera)
projector.project(particles, num_threads=8)

# Collapse the nested grids into the final 2D image array
image = projector.collapse()
```

---
## Advanced Configuration

The `Camera` class allows for fine-tuning the algorithm performance:
* **`target_cells_per_h` ($N_h$):** Controls the sampling quality. Higher values lead to smoother images but slightly slower projection.
* **`r_max`:** The maximum resolution level ($2^{r_{max}}$ pixels).
* **`periodic`:** Boolean to enable/disable periodic boundary conditions.
