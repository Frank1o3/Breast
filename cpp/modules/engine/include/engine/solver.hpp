#pragma once
// engine/solver.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Public API for the soft-body solver.
// This header is installed alongside libengine so that other modules
// (e.g. libcollision) can #include "engine/solver.hpp" and link libengine
// without duplicating any code.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <vector>

namespace softsim::engine {

// ─────────────────────────────────────────────────────────────────────────────
// Plain-data init structs  (used by Python bindings and other modules)
// ─────────────────────────────────────────────────────────────────────────────

struct PointInit {
    float x, y, z;
    bool  pinned = false;
};

struct SpringInit {
    int32_t a, b;
    float   stiffness   = 1.0f;
    float   rest_length = 0.0f;
};

// ─────────────────────────────────────────────────────────────────────────────
// Solver  — Jacobi soft-body with Verlet integration + pressure
// ─────────────────────────────────────────────────────────────────────────────
class Solver {
public:
    // ── Tuneable parameters (read/write from Python or C++) ─────────────────
    float stiffness          = 0.1f;
    float pressure_stiffness = 0.05f;
    float damping            = 0.999f;
    float floor_y            = -0.5f;
    float gravity_y          = -9.8f;
    int   iterations         = 6;

    // ── Read-only state ──────────────────────────────────────────────────────
    bool     is_exploded  = false;
    uint64_t steps_stable = 0;
    float    rest_volume  = 0.f;
    float    avg_edge     = 0.f;

    // ── Position / previous-position buffers ────────────────────────────────
    // Layout: flat float32 arrays of length N*3  (x0,y0,z0, x1,y1,z1, ...)
    // Exposed directly to bindings for zero-copy numpy access.
    std::vector<float> pos;
    std::vector<float> prev_pos;

    // ── Construction ────────────────────────────────────────────────────────
    Solver(
        const std::vector<PointInit>&  points,
        const std::vector<SpringInit>& springs,
        const std::vector<int32_t>&    faces_flat,  // F*3, row-major
        float gravity_y = -9.8f,
        float scale     = 1.0f
    );

    // ── Advance one sub-step ─────────────────────────────────────────────────
    void update(float dt);

    // ── Accessors ────────────────────────────────────────────────────────────
    int  num_points()  const;
    int  num_springs() const;
    int  num_faces()   const;

    // Write an (N*3) float32 array into pos directly (e.g. from shared memory)
    void set_positions(const float* data, int n);

private:
    // topology
    std::vector<int32_t> faces_;
    int n_faces_ = 0;

    // per-vertex
    std::vector<bool>    pinned_;
    std::vector<float>   pinned_pos_;

    // per-spring
    std::vector<int32_t> spring_a_, spring_b_;
    std::vector<float>   rest_len_, spring_k_;

    // work buffers (reused every step — no allocation in hot path)
    std::vector<float>   delta_;   // N*3
    std::vector<float>   counts_;  // N
    std::vector<float>   corr_;    // S*3  per-spring corrections

    // ── Internal kernels ─────────────────────────────────────────────────────
    void verlet_integrate_(float dt, float max_vel);
    void compute_spring_corrections_(int S);
    void scatter_spring_corrections_(int S);
    void apply_deltas_(int N);
    void apply_pressure_(float pval);
    void enforce_pinned_();
    float calc_volume_() const;
};

} // namespace softsim::engine
