/* -----------------------------------------------------------------------------
 * py-sphviewer2: High-performance SPH visualization.
 *
 * Copyright (c) 2024 Alejandro Benitez-Llambay
 * Distributed under the terms of the MIT License.
 * -----------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------------*/

#include <pybind11/pybind11.h>
#include "particle.hpp"
#include "smeshl.hpp" 
#include "smoothing_length.hpp" 

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    m.doc() = "sphviewer2 core C++ extension using pybind11";

    // Bind the SPHData struct
    py::class_<SPHData>(m, "SPHData")
        .def(py::init<>())
        .def_readwrite("x", &SPHData::x)
        .def_readwrite("y", &SPHData::y)
        .def_readwrite("z", &SPHData::z)
        .def_readwrite("h", &SPHData::h)
        .def_readwrite("m", &SPHData::m);

    // Bind the NestedGrids class and its methods
    py::class_<NestedGrids>(m, "NestedGrids")
        .def(
            py::init<double, int, int, int>(), 
            py::arg("box_size"), py::arg("target_cells_per_h"), py::arg("r_min"), py::arg("r_max")
        )
        .def(
            "project", &NestedGrids::project, 
            py::arg("particles"), py::arg("num_threads"),
            py::arg("extent"), py::arg("Lbox"), 
            py::arg("cam_xc"), py::arg("cam_yc"), py::arg("cam_zc"),
            py::arg("rot_matrix"),
            py::arg("periodic")
        )
        .def(
            "collapse", &NestedGrids::collapse, "Collapse grids and return the final high-res image"
        )
        .def(
            "get_level_data", &NestedGrids::get_level_data
        )
        .def(
            "set_level_data", &NestedGrids::set_level_data
        );

    // Bind the Helper
    m.def(
        "estimate_smoothing_length", &estimate_smoothing_length, 
        "Estimate SPH smoothing lengths using a Morton Z-curve",
        py::arg("x"), py::arg("y"), py::arg("z"), 
        py::arg("Lbox"), py::arg("k_neighbors") = 32,
        py::arg("num_threads") = 4
    );

}