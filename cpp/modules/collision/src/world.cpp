#include <cmath>
#include "collision/world.hpp"
#include "collision/narrow.hpp"

#ifdef COLLISION_OPENMP
#include <omp.h>
#endif

namespace collision
{

    // ── Add colliders ─────────────────────────────────────────────────────────────

    void World::add_plane(float nx, float ny, float nz, float d)
    {
        // Normalise the normal so resolve_plane math stays correct
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-8f)
        {
            planes_.push_back({nx / len, ny / len, nz / len, d / len});
        }
    }

    void World::add_sphere(float cx, float cy, float cz, float radius)
    {
        spheres_.push_back({cx, cy, cz, radius});
    }

    void World::add_box(float cx, float cy, float cz,
                        float hx, float hy, float hz)
    {
        boxes_.push_back({cx, cy, cz, hx, hy, hz});
    }

    void World::add_capsule(float ax, float ay, float az,
                            float bx, float by, float bz, float radius)
    {
        capsules_.push_back({ax, ay, az, bx, by, bz, radius});
    }

    // ── Move dynamic colliders ────────────────────────────────────────────────────

    void World::move_sphere(int idx, float cx, float cy, float cz)
    {
        spheres_[static_cast<size_t>(idx)].cx = cx;
        spheres_[static_cast<size_t>(idx)].cy = cy;
        spheres_[static_cast<size_t>(idx)].cz = cz;
    }

    void World::move_box(int idx, float cx, float cy, float cz)
    {
        boxes_[static_cast<size_t>(idx)].cx = cx;
        boxes_[static_cast<size_t>(idx)].cy = cy;
        boxes_[static_cast<size_t>(idx)].cz = cz;
    }

    void World::move_capsule(int idx,
                             float ax, float ay, float az,
                             float bx, float by, float bz)
    {
        auto &c = capsules_[static_cast<size_t>(idx)];
        c.ax = ax;
        c.ay = ay;
        c.az = az;
        c.bx = bx;
        c.by = by;
        c.bz = bz;
    }

    // ── Clear ─────────────────────────────────────────────────────────────────────

    void World::clear()
    {
        planes_.clear();
        spheres_.clear();
        boxes_.clear();
        capsules_.clear();
    }

    void World::clear_planes() { planes_.clear(); }
    void World::clear_spheres() { spheres_.clear(); }
    void World::clear_boxes() { boxes_.clear(); }
    void World::clear_capsules() { capsules_.clear(); }

    // ── Resolve ───────────────────────────────────────────────────────────────────

    void World::resolve(float *pos, const bool *pinned, int N, float friction) const
    {
#pragma omp parallel for schedule(static) if (N > 512)
        for (int i = 0; i < N; ++i)
        {
            if (pinned[i])
                continue;

            float *px = &pos[i * 3];
            float *py = &pos[i * 3 + 1];
            float *pz = &pos[i * 3 + 2];

            for (const auto &p : planes_)
                resolve_plane(px, py, pz, p, friction);

            for (const auto &s : spheres_)
                resolve_sphere(px, py, pz, s);

            for (const auto &b : boxes_)
                resolve_box(px, py, pz, b);

            for (const auto &c : capsules_)
                resolve_capsule(px, py, pz, c);
        }
    }

} // namespace collision
