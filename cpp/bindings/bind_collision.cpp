// bindings/bind_collision.cpp
// ─────────────────────────────────────────────────────────────────────────────
// pybind11 bindings for collision::World and collision::Broadphase.
//
// Exposed Python module: _collision
// ─────────────────────────────────────────────────────────────────────────────

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "collision/broadphase.hpp"
#include "collision/world.hpp"

namespace py = pybind11;
using namespace collision;

PYBIND11_MODULE(_collision, m)
{
    m.doc() = "collision — broadphase + narrow phase collision resolution";

    // ── World ─────────────────────────────────────────────────────────────────
    py::class_<World>(m, "World",
                      "Collision world containing static and dynamic colliders.\n\n"
                      "Call resolve() or use Broadphase.resolve_culled() each physics tick\n"
                      "BEFORE the spring and pressure solver steps.\n")
        .def(py::init<>())

        // ── Add colliders ─────────────────────────────────────────────────────────
        .def("add_plane",
             &World::add_plane,
             py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("d"),
             "Add an infinite plane. Normal (nx,ny,nz) is auto-normalised.\n"
             "Vertices on the side dot(n,p) >= d are considered outside.")

        .def("add_sphere",
             &World::add_sphere,
             py::arg("cx"), py::arg("cy"), py::arg("cz"), py::arg("radius"),
             "Add a solid sphere.")

        .def("add_box",
             &World::add_box,
             py::arg("cx"), py::arg("cy"), py::arg("cz"),
             py::arg("hx"), py::arg("hy"), py::arg("hz"),
             "Add an axis-aligned box. cx/cy/cz = centre, hx/hy/hz = half-extents.")

        .def("add_capsule",
             &World::add_capsule,
             py::arg("ax"), py::arg("ay"), py::arg("az"),
             py::arg("bx"), py::arg("by"), py::arg("bz"),
             py::arg("radius"),
             "Add a capsule defined by segment AB and a radius.")

        // ── Move dynamic colliders ────────────────────────────────────────────────
        .def("move_sphere",
             &World::move_sphere,
             py::arg("idx"), py::arg("cx"), py::arg("cy"), py::arg("cz"),
             "Update the centre of sphere at index idx.")

        .def("move_box",
             &World::move_box,
             py::arg("idx"), py::arg("cx"), py::arg("cy"), py::arg("cz"),
             "Update the centre of box at index idx.")

        .def("move_capsule",
             &World::move_capsule,
             py::arg("idx"),
             py::arg("ax"), py::arg("ay"), py::arg("az"),
             py::arg("bx"), py::arg("by"), py::arg("bz"),
             "Update the segment endpoints of capsule at index idx.")

        // ── Clear ─────────────────────────────────────────────────────────────────
        .def("clear", &World::clear, "Remove all colliders.")
        .def("clear_planes", &World::clear_planes)
        .def("clear_spheres", &World::clear_spheres)
        .def("clear_boxes", &World::clear_boxes)
        .def("clear_capsules", &World::clear_capsules)

        // ── Counts ────────────────────────────────────────────────────────────────
        .def("num_planes", &World::num_planes)
        .def("num_spheres", &World::num_spheres)
        .def("num_boxes", &World::num_boxes)
        .def("num_capsules", &World::num_capsules)

        // ── Resolve ───────────────────────────────────────────────────────────────
        .def("resolve", [](World &self, py::array_t<float, py::array::c_style> pos_arr, py::array_t<bool, py::array::c_style> pinned_arr, float friction)
             {
            auto rp = pos_arr.mutable_unchecked<2>();
            auto rn = pinned_arr.unchecked<1>();
            const int N = static_cast<int>(rp.shape(0));
            if (rn.shape(0) != N)
                throw std::runtime_error("resolve: pos and pinned length mismatch");
            self.resolve(&rp(0,0), &rn(0), N, friction); }, py::arg("pos"), py::arg("pinned"), py::arg("friction") = 0.1f, "Resolve all colliders against a (N,3) float32 position array.\n"
                                                                                 "pinned is a (N,) bool array — pinned vertices are skipped.\n"
                                                                                 "Modifies pos in-place.");

    // ── Broadphase ────────────────────────────────────────────────────────────
    py::class_<Broadphase>(m, "Broadphase",
                           "AABB broadphase wrapper around World.\n\n"
                           "Call rebuild() once per tick when colliders have moved,\n"
                           "then resolve_culled() instead of World.resolve() for large meshes.")
        .def(py::init<World &>(), py::arg("world"),
             py::keep_alive<1, 2>()) // Broadphase must not outlive World

        .def("rebuild", &Broadphase::rebuild,
             "Recompute per-collider AABBs. Call after moving any collider.")

        .def("resolve_culled", [](Broadphase &self, py::array_t<float, py::array::c_style> pos_arr, py::array_t<bool, py::array::c_style> pinned_arr, float friction)
             {
            auto rp = pos_arr.mutable_unchecked<2>();
            auto rn = pinned_arr.unchecked<1>();
            const int N = static_cast<int>(rp.shape(0));
            if (rn.shape(0) != N)
                throw std::runtime_error("resolve_culled: pos and pinned length mismatch");
            self.resolve_culled(&rp(0,0), &rn(0), N, friction); }, py::arg("pos"), py::arg("pinned"), py::arg("friction") = 0.1f, "Resolve with AABB broadphase cull — faster than World.resolve() for\n"
                                                                                 "large meshes with many colliders.");
}
