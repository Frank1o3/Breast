// engine/src/volume.cpp
#include "engine/volume.hpp"

#ifdef breast_OPENMP
#include <omp.h>
#endif

namespace breast::engine {

// Signed mesh volume via the divergence theorem.
// V = (1/6) * Σ  dot(v0, cross(v1, v2))
float calc_mesh_volume(
    const float*   pos,
    const int32_t* faces,
    int            n_faces
) {
    double total = 0.0;

    #pragma omp parallel for reduction(+:total) schedule(static) if(n_faces > 256)
    for (int i = 0; i < n_faces; ++i) {
        const int i0 = faces[i*3], i1 = faces[i*3+1], i2 = faces[i*3+2];

        const float x0 = pos[i0*3], y0 = pos[i0*3+1], z0 = pos[i0*3+2];
        const float x1 = pos[i1*3], y1 = pos[i1*3+1], z1 = pos[i1*3+2];
        const float x2 = pos[i2*3], y2 = pos[i2*3+1], z2 = pos[i2*3+2];

        total += static_cast<double>(
            x0 * (y1*z2 - y2*z1) +
            y0 * (z1*x2 - z2*x1) +
            z0 * (x1*y2 - x2*y1)
        );
    }

    return static_cast<float>(total < 0.0 ? -total : total) / 6.f;
}

} // namespace breast::engine
