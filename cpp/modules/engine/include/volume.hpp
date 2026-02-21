#pragma once

#include <cstdint>
#include <cmath>
#include <vector>

namespace breast::engine
{

    float calc_mesh_volume(
        const float *pos,
        const int32_t *faces,
        int n_faces);

    inline float calc_mesh_volume(
        const std::vector<float> &pos,
        const std::vector<int32_t> &faces_flat)
    {
        return calc_mesh_volume(pos.data(), faces_flat.data(), static_cast<int>(faces_flat.size() / 3));
    }

} // namespace breast::engine
