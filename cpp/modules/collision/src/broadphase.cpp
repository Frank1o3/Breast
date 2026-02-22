// collision/src/broadphase.cpp
#include "collision/broadphase.hpp"
#include "collision/narrow.hpp"

#ifdef COLLISION_OPENMP
#include <omp.h>
#endif

namespace collision
{

    // ── Rebuild AABBs ─────────────────────────────────────────────────────────────

    void Broadphase::rebuild()
    {
        const auto &pl = world_.planes();
        const auto &sp = world_.spheres();
        const auto &bx = world_.boxes();
        const auto &ca = world_.capsules();

        plane_aabbs_.resize(pl.size());
        for (size_t i = 0; i < pl.size(); ++i)
            plane_aabbs_[i] = aabb_of(pl[i]);

        sphere_aabbs_.resize(sp.size());
        for (size_t i = 0; i < sp.size(); ++i)
            sphere_aabbs_[i] = aabb_of(sp[i]);

        box_aabbs_.resize(bx.size());
        for (size_t i = 0; i < bx.size(); ++i)
            box_aabbs_[i] = aabb_of(bx[i]);

        capsule_aabbs_.resize(ca.size());
        for (size_t i = 0; i < ca.size(); ++i)
            capsule_aabbs_[i] = aabb_of(ca[i]);
    }

    // ── Resolve with broadphase cull ──────────────────────────────────────────────

    void Broadphase::resolve_culled(float *pos, const bool *pinned, int N,
                                    float friction) const
    {
        const auto &planes = world_.planes();
        const auto &spheres = world_.spheres();
        const auto &boxes = world_.boxes();
        const auto &capsules = world_.capsules();

#pragma omp parallel for schedule(static) if (N > 512)
        for (int i = 0; i < N; ++i)
        {
            if (pinned[i])
                continue;

            float *px = &pos[i * 3];
            float *py = &pos[i * 3 + 1];
            float *pz = &pos[i * 3 + 2];

            const float vx = *px, vy = *py, vz = *pz;

            // Planes — always test (infinite AABB, broadphase doesn't help here)
            for (size_t j = 0; j < planes.size(); ++j)
                resolve_plane(px, py, pz, planes[j], friction);

            // Spheres — skip if vertex is outside AABB
            for (size_t j = 0; j < spheres.size(); ++j)
            {
                if (!sphere_aabbs_[j].contains(vx, vy, vz))
                    continue;
                resolve_sphere(px, py, pz, spheres[j]);
            }

            // Boxes
            for (size_t j = 0; j < boxes.size(); ++j)
            {
                if (!box_aabbs_[j].contains(vx, vy, vz))
                    continue;
                resolve_box(px, py, pz, boxes[j]);
            }

            // Capsules
            for (size_t j = 0; j < capsules.size(); ++j)
            {
                if (!capsule_aabbs_[j].contains(vx, vy, vz))
                    continue;
                resolve_capsule(px, py, pz, capsules[j]);
            }
        }
    }

} // namespace collision
