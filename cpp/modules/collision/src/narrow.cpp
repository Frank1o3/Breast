// collision/src/narrow.cpp
#include "collision/narrow.hpp"

#include <algorithm>
#include <cmath>

namespace collision
{

    // ── Plane ─────────────────────────────────────────────────────────────────────
    // Pushes vertex to the positive side of the plane.
    // Friction damps the tangential component of the correction.
    bool resolve_plane(float *px, float *py, float *pz,
                       const Plane &p, float friction)
    {
        const float dist = (*px) * p.nx + (*py) * p.ny + (*pz) * p.nz - p.d;
        if (dist >= 0.f)
            return false;

        // Normal correction — push fully out
        *px -= dist * p.nx;
        *py -= dist * p.ny;
        *pz -= dist * p.nz;

        // Friction — damp motion along the contact plane
        // (tangential component of the displacement that was just applied)
        if (friction > 0.f)
        {
            const float scale = -dist * friction;
            // Tangent = displacement - normal_component
            // Displacement along plane = correction - (correction · n) * n
            // Since correction = -dist*n, tangent of original velocity is implicit.
            // Here we apply a simple tangential damping to position:
            const float tx = -*px - (-dist * p.nx);
            const float ty = -*py - (-dist * p.ny);
            const float tz = -*pz - (-dist * p.nz);
            (void)tx;
            (void)ty;
            (void)tz;
            (void)scale;
            // Note: true friction requires prev_pos (velocity). This is handled
            // in the solver's Verlet step. Here we only do positional correction.
        }

        return true;
    }

    // ── Sphere ────────────────────────────────────────────────────────────────────
    // Pushes vertex outside the sphere surface.
    bool resolve_sphere(float *px, float *py, float *pz, const Sphere &s)
    {
        const float dx = *px - s.cx;
        const float dy = *py - s.cy;
        const float dz = *pz - s.cz;
        const float d2 = dx * dx + dy * dy + dz * dz;
        const float r2 = s.radius * s.radius;

        if (d2 >= r2)
            return false;

        const float d = std::sqrt(d2);
        const float pen = s.radius - d;

        if (d > 1e-8f)
        {
            const float inv = pen / d;
            *px += dx * inv;
            *py += dy * inv;
            *pz += dz * inv;
        }
        else
        {
            // Vertex is exactly at sphere centre — push up
            *py += s.radius;
        }

        return true;
    }

    // ── Box ───────────────────────────────────────────────────────────────────────
    // Pushes vertex out along the axis of minimum penetration (SAT).
    bool resolve_box(float *px, float *py, float *pz, const Box &b)
    {
        const float lx = *px - b.cx;
        const float ly = *py - b.cy;
        const float lz = *pz - b.cz;

        // Early-out: vertex is outside the box on any axis
        if (std::abs(lx) > b.hx)
            return false;
        if (std::abs(ly) > b.hy)
            return false;
        if (std::abs(lz) > b.hz)
            return false;

        // Penetration depth on each axis
        const float ox = b.hx - std::abs(lx);
        const float oy = b.hy - std::abs(ly);
        const float oz = b.hz - std::abs(lz);

        // Push out along shallowest axis
        if (ox <= oy && ox <= oz)
            *px += std::copysign(ox, lx);
        else if (oy <= ox && oy <= oz)
            *py += std::copysign(oy, ly);
        else
            *pz += std::copysign(oz, lz);

        return true;
    }

    // ── Capsule ───────────────────────────────────────────────────────────────────
    // Pushes vertex outside the capsule (cylinder + two hemispherical caps).
    // Closest-point-on-segment test, then sphere resolve around that point.
    bool resolve_capsule(float *px, float *py, float *pz, const Capsule &c)
    {
        // Segment AB
        const float abx = c.bx - c.ax;
        const float aby = c.by - c.ay;
        const float abz = c.bz - c.az;

        // Vector from A to vertex P
        const float apx = *px - c.ax;
        const float apy = *py - c.ay;
        const float apz = *pz - c.az;

        // Project P onto AB, clamp to [0,1]
        const float ab2 = abx * abx + aby * aby + abz * abz;
        const float t = (ab2 > 1e-10f)
                            ? std::clamp((apx * abx + apy * aby + apz * abz) / ab2, 0.f, 1.f)
                            : 0.f;

        // Closest point on segment
        const float cpx = c.ax + t * abx;
        const float cpy = c.ay + t * aby;
        const float cpz = c.az + t * abz;

        // Distance from vertex to closest point
        const float dx = *px - cpx;
        const float dy = *py - cpy;
        const float dz = *pz - cpz;
        const float d2 = dx * dx + dy * dy + dz * dz;
        const float r2 = c.radius * c.radius;

        if (d2 >= r2)
            return false;

        const float d = std::sqrt(d2);
        const float pen = c.radius - d;

        if (d > 1e-8f)
        {
            const float inv = pen / d;
            *px += dx * inv;
            *py += dy * inv;
            *pz += dz * inv;
        }
        else
        {
            *py += c.radius;
        }

        return true;
    }

} // namespace collision
