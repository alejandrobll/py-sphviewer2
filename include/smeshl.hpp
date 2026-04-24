/* -----------------------------------------------------------------------------
 * py-sphviewer2: High-performance SPH visualization.
 *
 * Copyright (c) 2024 Alejandro Benitez-Llambay
 * Distributed under the terms of the MIT License.
 * -----------------------------------------------------------------------------
 * Implementation of the main rendering algorithm as described in Benitez-Llambay (2025)
 * ---------------------------------------------------------------------------*/

#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <thread>
#include <mutex>
#include "particle.hpp"

// Custom kernel, converted to a type-safe inline function
inline double evaluate_kernel(double R2, double hinv2) {
    // 2.5 / pi is the analytical 2D normalization for (1 - (r/h)^2)^1.5
    constexpr double PI = 3.14159265358979323846;
    constexpr double norm_2d = 2.5 / PI; 
    
    return (R2 < 1.0) ? (norm_2d * hinv2 * std::pow(1.0 - R2, 1.5)) : 0.0;
}

// Structure to of a single grid at a specific resolution level
struct GridLevel {
    int R;                  
    int N_cells;            
    double delta;           
    std::vector<double> data; 

    GridLevel(int level, double L) : R(level) {
        // A grid at resolution level R has 2^R number of cells per dimension 
        N_cells = 1 << R; // Equivalent to 2^R, but faster bitwise shift
        delta = L / N_cells;
        
        // Allocate memory for the 2D grid and initialize to 0
        data.resize(N_cells * N_cells, 0.0);
    }
};

// A simple struct to pass the 3x3 matrix to C++
struct ProjectionMatrix {
    double m[3][3];
    double center[3];
};

// Manager class to handle the hierarchy of nested grids
class NestedGrids {
private:
    std::vector<GridLevel> levels;
    double L;               
    int N_h;                
    int R_min;              
    int R_max;              
    std::mutex merge_mutex; 

public:
    NestedGrids(double box_size, int target_cells_per_h, int r_min, int r_max) 
        : L(box_size), N_h(target_cells_per_h), R_min(r_min), R_max(r_max) {
        
        // Initialize the sequence of nested grids from R_min to R_max
        for (int r = R_min; r <= R_max; ++r) {
            levels.emplace_back(r, L);
        }
    }

    // Determine the "native" resolution level for a particle
    int get_native_level(double h_j) const {
        int R_j = static_cast<int>(std::floor(std::log2(N_h) - std::log2(h_j / L)));
        
        // Clamp the level between R_min and R_max to prevent out-of-bounds access
        if (R_j < R_min) return R_min;
        if (R_j > R_max) return R_max;
        return R_j;
    }

    // Access a specific grid level by its R value
    GridLevel* get_grid(int R_j) {
        if (R_j >= R_min && R_j <= R_max) {
            return &levels[R_j - R_min];
        }
        return nullptr;
    }
    
    // project takes care of projecting the particles
    void project(const SPHData& particles, int num_threads, 
                 double extent, double Lbox, 
                 double cam_xc, double cam_yc, double cam_zc,
                 py::array_t<double> rot_matrix,
                 bool periodic) {
        
        int num_particles = particles.x.request().size;

        // --- SAFETY CHECKS ---
        if (particles.y.request().size != num_particles ||
            particles.z.request().size != num_particles ||
            particles.h.request().size != num_particles ||
            particles.m.request().size != num_particles) {
            throw std::runtime_error("SPHData size mismatch! Did you forget to assign all arrays (x, y, z, h, m) in Python?");
        }  
        if (rot_matrix.request().size != 9) {
            throw std::runtime_error("Rotation matrix must be exactly 9 elements!");
        }
        // ------------------

        auto ptr_x = static_cast<double*>(particles.x.request().ptr);
        auto ptr_y = static_cast<double*>(particles.y.request().ptr);
        auto ptr_z = static_cast<double*>(particles.z.request().ptr);
        auto ptr_h = static_cast<double*>(particles.h.request().ptr);
        auto ptr_m = static_cast<double*>(particles.m.request().ptr);

        // Get the rotation matrix values
        auto R = static_cast<double*>(rot_matrix.request().ptr);

        auto worker = [&](int start_idx, int end_idx) {
            std::vector<GridLevel> local_levels = this->levels;
            for(auto& level : local_levels) {
                std::fill(level.data.begin(), level.data.end(), 0.0);
            }

            for (int i = start_idx; i < end_idx; ++i) {
                double h = ptr_h[i];
                if (h <= 0.0) continue; 

                // We do not allow h to be larger than half of the box
                if (h > Lbox / 2.0){
                    h = Lbox / 2.0;
                }

                // Distance from camera center
                double dx = ptr_x[i] - cam_xc;
                double dy = ptr_y[i] - cam_yc;
                double dz = ptr_z[i] - cam_zc;

                // Global Periodic Wrapping
                if (periodic) {
                    dx -= Lbox * std::round(dx / Lbox);
                    dy -= Lbox * std::round(dy / Lbox);
                    dz -= Lbox * std::round(dz / Lbox);                    
                }

                // Apply 3D Rotation Matrix
                // Rotated X' and Y' are our new "on-screen" coordinates
                // Matrix multiplication: P' = R * P
                double rx = R[0] * dx + R[1] * dy + R[2] * dz;
                double ry = R[3] * dx + R[4] * dy + R[5] * dz;

                // Shift to Camera Local Space [0, extent]
                double loc_x = rx + extent / 2.0;
                double loc_y = ry + extent / 2.0;

                // Skip if particle is completely off-camera
                if (loc_x + h < 0.0 || loc_x - h > extent || 
                    loc_y + h < 0.0 || loc_y - h > extent) {
                    continue; 
                }

                int R_j = get_native_level(h);
                int level_idx = R_j - R_min;
                GridLevel& grid = local_levels[level_idx];

                double delta = grid.delta;
                int N_cells = grid.N_cells;
                double hinv2 = 1.0 / (h * h);

                int min_ix = static_cast<int>(std::floor((loc_x - h) / delta));
                int max_ix = static_cast<int>(std::floor((loc_x + h) / delta));
                int min_iy = static_cast<int>(std::floor((loc_y - h) / delta));
                int max_iy = static_cast<int>(std::floor((loc_y + h) / delta));

                for (int cy = min_iy; cy <= max_iy; ++cy) {
                    double cell_y = (cy + 0.5) * delta;
                    double dist_y = cell_y - loc_y;
                    double dy2 = dist_y * dist_y;

                    for (int cx = min_ix; cx <= max_ix; ++cx) {
                        double cell_x = (cx + 0.5) * delta;
                        double dist_x = cell_x - loc_x;
                        double R2 = (dist_x * dist_x + dy2) * hinv2;
                        
                        double w = evaluate_kernel(R2, hinv2);
                        
                        if (w > 0.0) {
                            // Open Boundaries: Just check if inside the image!
                            if (cx >= 0 && cx < N_cells && cy >= 0 && cy < N_cells) {
                                grid.data[cy * N_cells + cx] += ptr_m[i] * w;
                            }
                        }
                    }
                }
            }
            // Once the thread finishes its chunk, lock the mutex and merge 
            // its local grids into the global master grids.
            std::lock_guard<std::mutex> lock{this->merge_mutex};
            for (size_t lvl = 0; lvl < this->levels.size(); ++lvl) {
                for (size_t k = 0; k < this->levels[lvl].data.size(); ++k) {
                    this->levels[lvl].data[k] += local_levels[lvl].data[k];
                }
            }
        };

        // Chunk the particles and launch the threads
        std::vector<std::thread> threads;
        int chunk_size = num_particles / num_threads;
        
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? num_particles : (start + chunk_size);
            threads.emplace_back(worker, start, end);
        }

        // Wait for all threads to finish projecting
        for (auto& t : threads) {
            t.join();
        }
    }             

    py::array_t<double> collapse() {
        // Cascade from the lowest resolution (0) to the highest (size - 2)
        for (size_t lvl = 0; lvl < levels.size() - 1; ++lvl) {
            GridLevel& coarse = levels[lvl];
            GridLevel& fine = levels[lvl + 1];

            int Nc = coarse.N_cells;
            int Nf = fine.N_cells; // This is strictly 2 * Nc

            // Bilinear interpolation from coarse to fine
            for (int fy = 0; fy < Nf; ++fy) {
                // Calculate the exact floating-point coordinate in the coarse grid
                double cy_exact = (fy + 0.5) / 2.0 - 0.5;
                int cy0 = static_cast<int>(std::floor(cy_exact));
                int cy1 = cy0 + 1;
                
                // Interpolation weights
                double wy1 = cy_exact - cy0;
                double wy0 = 1.0 - wy1;

                // Fast periodic wrap for the coarse Y indices
                int cy0_c = std::max(0, std::min(cy0, Nc - 1));
                int cy1_c = std::max(0, std::min(cy1, Nc - 1));

                for (int fx = 0; fx < Nf; ++fx) {
                    double cx_exact = (fx + 0.5) / 2.0 - 0.5;
                    int cx0 = static_cast<int>(std::floor(cx_exact));
                    int cx1 = cx0 + 1;

                    int cx0_c = std::max(0, std::min(cx0, Nc - 1));
                    int cx1_c = std::max(0, std::min(cx1, Nc - 1));                    

                    double wx1 = cx_exact - cx0;
                    double wx0 = 1.0 - wx1;

                    double interpolated_val = 
                        wy0 * wx0 * coarse.data[cy0_c * Nc + cx0_c] +
                        wy0 * wx1 * coarse.data[cy0_c * Nc + cx1_c] +
                        wy1 * wx0 * coarse.data[cy1_c * Nc + cx0_c] +
                        wy1 * wx1 * coarse.data[cy1_c * Nc + cx1_c];                        

                    // Add the interpolated coarse data to the fine grid!
                    fine.data[fy * Nf + fx] += interpolated_val;
                }
            }
        }

        // The highest resolution grid now contains all the condensed information
        GridLevel& highest = levels.back();
        int N_high = highest.N_cells;

        // Allocate a 2D NumPy array to return to Python
        py::array_t<double> result({N_high, N_high});
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);

        // Copy the final data into the NumPy array buffer
        std::copy(highest.data.begin(), highest.data.end(), ptr);

        return result;
    }

    // Helper functions for interpreting baked images
    py::array_t<double> get_level_data(int R_j) {
        int idx = R_j - R_min;
        if (idx < 0 || idx >= static_cast<int>(levels.size())) {
            throw std::runtime_error("Resolution level R_j out of bounds.");
        }
        
        int N = levels[idx].N_cells;
        py::array_t<double> result({N, N});
        double* ptr = static_cast<double*>(result.request().ptr);
        
        // Copy internal vector to NumPy
        std::copy(levels[idx].data.begin(), levels[idx].data.end(), ptr);
        return result;
    }

    void set_level_data(int R_j, py::array_t<double> input) {
        int idx = R_j - R_min;
        if (idx < 0 || idx >= static_cast<int>(levels.size())) {
            throw std::runtime_error("Resolution level R_j out of bounds.");
        }
        
        auto buf = input.request();
        if (buf.size != static_cast<py::ssize_t>(levels[idx].data.size())) {
            throw std::runtime_error("Input array size mismatch for this level.");
        }
        
        double* ptr = static_cast<double*>(buf.ptr);
        // Copy NumPy back to internal vector
        std::copy(ptr, ptr + buf.size, levels[idx].data.begin());
    }


};