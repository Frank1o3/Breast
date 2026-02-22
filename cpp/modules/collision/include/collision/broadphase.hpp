#pragma once
// collision/broadphase.hpp
// ─────────────────────────────────────────────────────────────────────────────
// AABB broadphase — quickly reject vertices that can't possibly touch any
// collider, so narrow phase only runs on plausible candidates.
//
// Usage is optional: World::resolve() works without it, but for large meshes
// (10k+ vertices) wrapping resolve() with a broadphase cull is worthwhile.
// ─────────────────────────────────────────────────────────────────────────────

#include "collision/world.hpp"
#include <vector>

namespace collision {

// ── Axis-aligned bounding box ─────────────────────────────────────────────────

struct AABB {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;

    bool overlaps(const AABB& o) const {
        return min_x <= o.max_x && max_x >= o.min_x
            && min_y <= o.max_y && max_y >= o.min_y
            && min_z <= o.max_z && max_z >= o.min_z;
    }

    bool contains(float x, float y, float z) const {
        return x >= min_x && x <= max_x
            && y >= min_y && y <= max_y
            && z >= min_z && z <= max_z;
    }
};

// ── Shape → AABB helpers ──────────────────────────────────────────────────────

inline AABB aabb_of(const Plane&, float world_half = 1000.f) {
    // Planes are infinite — return a large world AABB
    return { -world_half, -world_half, -world_half,
              world_half,  world_half,  world_half };
}

inline AABB aabb_of(const Sphere& s) {
    return { s.cx - s.radius, s.cy - s.radius, s.cz - s.radius,
             s.cx + s.radius, s.cy + s.radius, s.cz + s.radius };
}

inline AABB aabb_of(const Box& b) {
    return { b.cx - b.hx, b.cy - b.hy, b.cz - b.hz,
             b.cx + b.hx, b.cy + b.hy, b.cz + b.hz };
}

inline AABB aabb_of(const Capsule& c) {
    float r  = c.radius;
    float mnx = std::min(c.ax, c.bx) - r;
    float mny = std::min(c.ay, c.by) - r;
    float mnz = std::min(c.az, c.bz) - r;
    float mxx = std::max(c.ax, c.bx) + r;
    float mxy = std::max(c.ay, c.by) + r;
    float mxz = std::max(c.az, c.bz) + r;
    return { mnx, mny, mnz, mxx, mxy, mxz };
}

// ── Broadphase ────────────────────────────────────────────────────────────────
// Builds per-collider AABBs once per frame (call rebuild() when colliders
// move), then for each vertex returns only the colliders whose AABB contains
// that vertex.

class Broadphase {
public:
    explicit Broadphase(const World& world) : world_(world) {}

    // Recompute AABBs — call once per physics tick before resolve_culled().
    void rebuild();

    // Resolve only against colliders whose AABB overlaps the vertex.
    // Signature matches World::resolve() so it's a drop-in replacement.
    void resolve_culled(float* pos, const bool* pinned, int N,
                        float friction = 0.1f) const;

    // Per-shape AABBs (read-only, available after rebuild())
    const std::vector<AABB>& plane_aabbs()   const { return plane_aabbs_;   }
    const std::vector<AABB>& sphere_aabbs()  const { return sphere_aabbs_;  }
    const std::vector<AABB>& box_aabbs()     const { return box_aabbs_;     }
    const std::vector<AABB>& capsule_aabbs() const { return capsule_aabbs_; }

private:
    const World& world_;

    std::vector<AABB> plane_aabbs_;
    std::vector<AABB> sphere_aabbs_;
    std::vector<AABB> box_aabbs_;
    std::vector<AABB> capsule_aabbs_;
};

} // namespace collision
