#pragma once
// collision/narrow.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Per-shape contact resolution functions.
// These are free functions — no state, no allocation.
// Called by world.cpp and broadphase.cpp.
// ─────────────────────────────────────────────────────────────────────────────

#include "collision/world.hpp"

namespace collision
{

    // Each function takes a pointer to a single vertex's x/y/z components and
    // modifies them in-place if the vertex penetrates the shape.
    // Returns true if a contact was resolved, false if the vertex was outside.

    bool resolve_plane(float *px, float *py, float *pz,
                       const Plane &p, float friction);

    bool resolve_sphere(float *px, float *py, float *pz,
                        const Sphere &s);

    bool resolve_box(float *px, float *py, float *pz,
                     const Box &b);

    bool resolve_capsule(float *px, float *py, float *pz,
                         const Capsule &c);

} // namespace collision
