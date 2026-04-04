/* -----------------------------------------------------------------------------
 * py-sphviewer2: High-performance SPH visualization.
 *
 * Copyright (c) 2024 Alejandro Benitez-Llambay
 * Distributed under the terms of the MIT License.
 * -----------------------------------------------------------------------------
 * Definition of the particle data structure
 * ---------------------------------------------------------------------------*/

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Structure to hold the SPH data passed directly from Python to avoid copy of memory 
struct SPHData {
    py::array_t<double> x;
    py::array_t<double> y;
    py::array_t<double> z;
    py::array_t<double> h; 
    py::array_t<double> m; 
};