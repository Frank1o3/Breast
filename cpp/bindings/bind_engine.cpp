// bindings/bind_engine.cpp
// ─────────────────────────────────────────────────────────────────────────────
// pybind11 bindings for softsim::engine.
//
// This file is intentionally thin — no physics logic lives here.
// All logic is in libengine (modules/engine/src/).
//
// Exposed Python module: _engine
// Import:  import _engine
// ─────────────────────────────────────────────────────────────────────────────

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine/solver.hpp"
#include "engine/volume.hpp"

namespace py = pybind11;
using namespace breast::engine;

// ─────────────────────────────────────────────────────────────────────────────
// Conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

// points_arr  shape (N, 4)  float32:  x  y  z  pinned(0/1)
static std::vector<PointInit>
parse_points(py::array_t<float, py::array::c_style> arr) {
    auto r = arr.unchecked<2>();
    const int N = static_cast<int>(r.shape(0));
    std::vector<PointInit> out(N);
    for (int i = 0; i < N; ++i)
        out[i] = { r(i,0), r(i,1), r(i,2), r(i,3) > 0.5f };
    return out;
}

// spring_idx  shape (S, 2)  int32:   a  b
// spring_f    shape (S, 2)  float32: stiffness  rest_length
static std::vector<SpringInit>
parse_springs(py::array_t<int32_t, py::array::c_style> idx_arr,
              py::array_t<float,   py::array::c_style> f_arr) {
    auto ri = idx_arr.unchecked<2>();
    auto rf = f_arr.unchecked<2>();
    const int S = static_cast<int>(ri.shape(0));
    std::vector<SpringInit> out(S);
    for (int s = 0; s < S; ++s)
        out[s] = { ri(s,0), ri(s,1), rf(s,0), rf(s,1) };
    return out;
}

// faces_arr  shape (F, 3)  int32
static std::vector<int32_t>
parse_faces(py::array_t<int32_t, py::array::c_style> arr) {
    auto r = arr.unchecked<2>();
    const int F = static_cast<int>(r.shape(0));
    std::vector<int32_t> flat(F * 3);
    for (int i = 0; i < F; ++i) {
        flat[i*3]   = r(i,0);
        flat[i*3+1] = r(i,1);
        flat[i*3+2] = r(i,2);
    }
    return flat;
}

// ─────────────────────────────────────────────────────────────────────────────
// Module
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(_engine, m) {
    m.doc() = "softsim engine — soft-body physics (C++ / OpenMP)";

    // ── PointInit ─────────────────────────────────────────────────────────────
    py::class_<PointInit>(m, "PointInit")
        .def(py::init<>())
        .def_readwrite("x",      &PointInit::x)
        .def_readwrite("y",      &PointInit::y)
        .def_readwrite("z",      &PointInit::z)
        .def_readwrite("pinned", &PointInit::pinned)
        .def("__repr__", [](const PointInit& p) {
            return "<PointInit x=" + std::to_string(p.x)
                 + " y=" + std::to_string(p.y)
                 + " z=" + std::to_string(p.z)
                 + " pinned=" + (p.pinned ? "True" : "False") + ">";
        });

    // ── SpringInit ────────────────────────────────────────────────────────────
    py::class_<SpringInit>(m, "SpringInit")
        .def(py::init<>())
        .def_readwrite("a",           &SpringInit::a)
        .def_readwrite("b",           &SpringInit::b)
        .def_readwrite("stiffness",   &SpringInit::stiffness)
        .def_readwrite("rest_length", &SpringInit::rest_length)
        .def("__repr__", [](const SpringInit& s) {
            return "<SpringInit a=" + std::to_string(s.a)
                 + " b=" + std::to_string(s.b)
                 + " k=" + std::to_string(s.stiffness)
                 + " L=" + std::to_string(s.rest_length) + ">";
        });

    // ── Solver ────────────────────────────────────────────────────────────────
    py::class_<Solver>(m, "Solver",
        "Jacobi soft-body solver with Verlet integration and pressure.\n\n"
        "Constructor arrays\n"
        "------------------\n"
        "points_array   (N,4) float32 : x  y  z  pinned\n"
        "spring_indices (S,2) int32   : vertex_a  vertex_b\n"
        "spring_floats  (S,2) float32 : stiffness  rest_length\n"
        "faces_array    (F,3) int32   : v0  v1  v2\n"
    )
    .def(py::init([](
        py::array_t<float,   py::array::c_style> pts_arr,
        py::array_t<int32_t, py::array::c_style> sp_idx,
        py::array_t<float,   py::array::c_style> sp_f,
        py::array_t<int32_t, py::array::c_style> faces_arr,
        float gravity_y,
        float scale
    ) {
        return std::make_unique<Solver>(
            parse_points(pts_arr),
            parse_springs(sp_idx, sp_f),
            parse_faces(faces_arr),
            gravity_y,
            scale
        );
    }),
    py::arg("points_array"),
    py::arg("spring_indices"),
    py::arg("spring_floats"),
    py::arg("faces_array"),
    py::arg("gravity_y") = -9.8f,
    py::arg("scale")     = 1.0f)

    // ── Tuneable parameters ───────────────────────────────────────────────────
    .def_readwrite("stiffness",          &Solver::stiffness)
    .def_readwrite("pressure_stiffness", &Solver::pressure_stiffness)
    .def_readwrite("damping",            &Solver::damping)
    .def_readwrite("floor_y",            &Solver::floor_y)
    .def_readwrite("gravity_y",          &Solver::gravity_y)
    .def_readwrite("iterations",         &Solver::iterations)

    // ── Read-only state ───────────────────────────────────────────────────────
    .def_readonly("is_exploded",   &Solver::is_exploded)
    .def_readonly("steps_stable",  &Solver::steps_stable)
    .def_readonly("rest_volume",   &Solver::rest_volume)
    .def_readonly("avg_edge",      &Solver::avg_edge)
    .def("num_points",  &Solver::num_points)
    .def("num_springs", &Solver::num_springs)
    .def("num_faces",   &Solver::num_faces)

    // ── Main update ───────────────────────────────────────────────────────────
    .def("update", &Solver::update, py::arg("dt"),
         "Advance simulation by dt seconds (one sub-step).")

    // ── Zero-copy position access ─────────────────────────────────────────────
    // get_pos() returns a numpy view directly into the C++ vector — no copy.
    .def("get_pos", [](Solver& self) -> py::array_t<float> {
        const int N = self.num_points();
        return py::array_t<float>(
            { N, 3 },
            { 3 * sizeof(float), sizeof(float) },
            self.pos.data(),
            py::cast(self)   // keep-alive: array cannot outlive Solver
        );
    }, "Return zero-copy (N,3) float32 view of current positions.")

    .def("set_pos", [](Solver& self,
                       py::array_t<float, py::array::c_style> arr) {
        auto r = arr.unchecked<2>();
        if (r.shape(0) != self.num_points() || r.shape(1) != 3)
            throw std::runtime_error("set_pos: expected shape (N,3)");
        std::copy(arr.data(), arr.data() + self.num_points() * 3,
                  self.pos.begin());
    }, "Overwrite positions with a (N,3) float32 array.");

    // ── Free functions ────────────────────────────────────────────────────────
    m.def("calc_mesh_volume",
        [](py::array_t<float,   py::array::c_style> pos_arr,
           py::array_t<int32_t, py::array::c_style> faces_arr) -> float {
            auto rp = pos_arr.unchecked<2>();
            auto rf = faces_arr.unchecked<2>();
            const int F = static_cast<int>(rf.shape(0));
            std::vector<int32_t> flat(F * 3);
            for (int i = 0; i < F; ++i) {
                flat[i*3]   = rf(i,0);
                flat[i*3+1] = rf(i,1);
                flat[i*3+2] = rf(i,2);
            }
            return calc_mesh_volume(
                pos_arr.data(), flat.data(), F
            );
        },
        py::arg("pos"),
        py::arg("faces"),
        "Compute signed mesh volume (divergence theorem)."
    );
}
