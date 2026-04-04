/* -----------------------------------------------------------------------------
 * py-sphviewer2: High-performance SPH visualization.
 *
 * Copyright (c) 2024 Alejandro Benitez-Llambay
 * Distributed under the terms of the MIT License.
 * -----------------------------------------------------------------------------
 * Helper function to efficiently calculate the SPH smoothing length.
 * ---------------------------------------------------------------------------*/

#pragma once

#include <vector>
#include <cmath>
#include <queue>
#include <thread>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Array of Structures for maximum Cache Locality
struct PackedParticle {
    double x, y, z;
    int orig_idx;
    int cell_idx;

    // Sort ascending by cell index
    bool operator<(const PackedParticle& other) const {
        return cell_idx < other.cell_idx;
    }
};

// Estimates the smoothing length of the particles given a desired number of neighbors
inline py::array_t<double> estimate_smoothing_length(
    py::array_t<double> py_x, 
    py::array_t<double> py_y, 
    py::array_t<double> py_z, 
    double Lbox, 
    int k_neighbors,
    int num_threads) 
{
    auto req_x = py_x.request();
    int N = req_x.size;
    double* x = static_cast<double*>(req_x.ptr);
    double* y = static_cast<double*>(py_y.request().ptr);
    double* z = static_cast<double*>(py_z.request().ptr);

    py::array_t<double> py_h(N);
    double* h = static_cast<double*>(py_h.request().ptr);

    // Dynamic Grid Definition
    // We cap at 256^3 (16.7 million cells) to keep the offset arrays lean
    int M = std::min(256, std::max(64, static_cast<int>(std::cbrt(N) * 1.5))); 
    double cell_size = Lbox / M;
    double inv_cell_size = 1.0 / cell_size;
    int total_cells = M * M * M;

    std::vector<PackedParticle> particles(N);

    // Populate particles
    for (int i = 0; i < N; ++i) {
        double px = x[i];
        double py = y[i];
        double pz = z[i];

        // Safe modulo wrapping for [0, M-1]
        int cx = static_cast<int>(std::floor(px * inv_cell_size)) % M;
        int cy = static_cast<int>(std::floor(py * inv_cell_size)) % M;
        int cz = static_cast<int>(std::floor(pz * inv_cell_size)) % M;
        if (cx < 0) cx += M;
        if (cy < 0) cy += M;
        if (cz < 0) cz += M;

        particles[i] = {px, py, pz, i, (cz * M + cy) * M + cx};
    }

    // Sort the particles to ensure contiguous memory. This should be (O(N log N), so very fast
    std::sort(particles.begin(), particles.end());

    // Build the Offset Lookups (O(N))
    std::vector<int> cell_start(total_cells, -1);
    std::vector<int> cell_end(total_cells, -1);

    for (int i = 0; i < N; ++i) {
        int c = particles[i].cell_idx;
        if (i == 0 || c != particles[i - 1].cell_idx) {
            cell_start[c] = i;
        }
        if (i == N - 1 || c != particles[i + 1].cell_idx) {
            cell_end[c] = i + 1;
        }
    }

    // Parallel Search
    int chunk_size = (N + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    auto search_worker = [&](int start_idx, int end_idx) {
        // Notice we loop over the SORTED array. 
        // This means thread 1 is processing particles physically grouped together
        for (int i = start_idx; i < end_idx; ++i) {
            const auto& p = particles[i];
            
            int cx0 = p.cell_idx % M;
            int cy0 = (p.cell_idx / M) % M;
            int cz0 = p.cell_idx / (M * M);

            std::priority_queue<std::pair<double, int>> pq;
            int radius_cells = 0;
            bool search_done = false;

            while (!search_done) {
                // Loop over the outer shell boundaries
                for (int dz = -radius_cells; dz <= radius_cells; ++dz) {
                    for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
                        for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
                            
                            // Only check the newly expanded outer shell
                            if (radius_cells > 0 && std::abs(dx) != radius_cells && 
                                std::abs(dy) != radius_cells && std::abs(dz) != radius_cells) {
                                continue; 
                            }

                            int ncx = (cx0 + dx % M + M) % M;
                            int ncy = (cy0 + dy % M + M) % M;
                            int ncz = (cz0 + dz % M + M) % M;
                            int target_cell = (ncz * M + ncy) * M + ncx;

                            int start = cell_start[target_cell];
                            int end = cell_end[target_cell];

                            if (start != -1) {
                                // This should be a zero pointer chasing, so the CPU should be happy with it.
                                for (int j = start; j < end; ++j) {
                                    if (i == j) continue;
                                    
                                    const auto& np = particles[j];
                                    double dist_x = p.x - np.x;
                                    double dist_y = p.y - np.y;
                                    double dist_z = p.z - np.z;

                                    dist_x -= Lbox * std::round(dist_x / Lbox);
                                    dist_y -= Lbox * std::round(dist_y / Lbox);
                                    dist_z -= Lbox * std::round(dist_z / Lbox);

                                    double dist_sq = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;

                                    if (pq.size() < static_cast<size_t>(k_neighbors)) {
                                        pq.push({dist_sq, j});
                                    } else if (dist_sq < pq.top().first) {
                                        pq.pop();
                                        pq.push({dist_sq, j});
                                    }
                                }
                            }
                        }
                    }
                }

                // Strict stopping condition, so we do not have artifacts
                double search_phys_rad = radius_cells * cell_size;
                if (pq.size() == static_cast<size_t>(k_neighbors) && 
                    pq.top().first <= search_phys_rad * search_phys_rad) {
                    search_done = true;
                } else {
                    radius_cells++;
                    if (radius_cells > M / 2) search_done = true; // Fallback for empty boxes
                }
            }

            // Map the result back to the original index layout
            if (pq.size() == static_cast<size_t>(k_neighbors)) {
                h[p.orig_idx] = std::sqrt(pq.top().first);
            } else {
                h[p.orig_idx] = 0.0; 
            }
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, N);
        if (start < N) {
            threads.emplace_back(search_worker, start, end);
        }
    }

    // Wait for all threads to finish
    for (auto& t : threads) t.join();

    return py_h;
}