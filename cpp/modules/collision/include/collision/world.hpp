#pragma once
// collision/world.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Standalone collision world — no dependency on engine or any other module.
// Works on any flat float32 position buffer (x0 y0 z0 x1 y1 z1 ...).
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <vector>

namespace collision {

// ── Collider shapes ───────────────────────────────────────────────────────────

struct Plane {
    float nx, ny, nz;   // unit outward normal
    float d;            // offset: dot(n, p) >= d means outside
};

struct Sphere {
    float cx, cy, cz;
    float radius;
};

struct Box {
    float cx, cy, cz;   // centre
    float hx, hy, hz;   // half-extents (all positive)
};

struct Capsule {
    float ax, ay, az;   // segment start
    float bx, by, bz;   // segment end
    float radius;
};

// ── Collision world ───────────────────────────────────────────────────────────

class World {
public:
    // ── Add static colliders ──────────────────────────────────────────────────
    void add_plane  (float nx, float ny, float nz, float d);
    void add_sphere (float cx, float cy, float cz, float radius);
    void add_box    (float cx, float cy, float cz,
                     float hx, float hy, float hz);
    void add_capsule(float ax, float ay, float az,
                     float bx, float by, float bz, float radius);

    // ── Update dynamic colliders ──────────────────────────────────────────────
    // Index matches the order they were added via add_sphere / add_box / add_capsule.
    void move_sphere (int idx, float cx, float cy, float cz);
    void move_box    (int idx, float cx, float cy, float cz);
    void move_capsule(int idx,
                      float ax, float ay, float az,
                      float bx, float by, float bz);

    // ── Remove all colliders ──────────────────────────────────────────────────
    void clear();
    void clear_planes();
    void clear_spheres();
    void clear_boxes();
    void clear_capsules();

    // ── Counts ────────────────────────────────────────────────────────────────
    int num_planes()   const { return static_cast<int>(planes_.size());   }
    int num_spheres()  const { return static_cast<int>(spheres_.size());  }
    int num_boxes()    const { return static_cast<int>(boxes_.size());    }
    int num_capsules() const { return static_cast<int>(capsules_.size()); }

    // ── Main resolve pass ─────────────────────────────────────────────────────
    // Pushes vertices out of all registered colliders.
    // Call BEFORE spring and pressure steps each physics tick.
    //
    // pos     — flat float32, length N*3  (x0 y0 z0 x1 y1 z1 ...)
    // pinned  — bool mask, length N       (pinned vertices are skipped)
    // N       — vertex count
    // friction — [0,1]  0 = frictionless, 1 = full stick along contact tangent
    void resolve(float* pos, const bool* pinned, int N, float friction = 0.1f) const;

    // ── Direct shape accessors (read-only) ────────────────────────────────────
    const std::vector<Plane>&   planes()   const { return planes_;   }
    const std::vector<Sphere>&  spheres()  const { return spheres_;  }
    const std::vector<Box>&     boxes()    const { return boxes_;    }
    const std::vector<Capsule>& capsules() const { return capsules_; }

private:
    std::vector<Plane>   planes_;
    std::vector<Sphere>  spheres_;
    std::vector<Box>     boxes_;
    std::vector<Capsule> capsules_;
};

} // namespace collision
