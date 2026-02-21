#pragma once
// engine/volume.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Stand-alone volume / area utilities used by Solver and exposed to bindings.
// Kept in its own header so other modules (e.g. a collision module) can use
// them without pulling in the full Solver.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cmath>
#include <vector>

namespace softsim::engine {

// Signed mesh volume via divergence theorem.
// pos       — flat float32 array, length N*3
// faces     — flat int32  array, length F*3
float calc_mesh_volume(
    const float*   pos,
    const int32_t* faces,
    int            n_faces
);

// Convenience overload for std::vector inputs.
inline float calc_mesh_volume(
    const std::vector<float>&   pos,
    const std::vector<int32_t>& faces_flat
) {
    return calc_mesh_volume(
        pos.data(),
        faces_flat.data(),
        static_cast<int>(faces_flat.size()) / 3
    );
}

} // namespace softsim::engine
