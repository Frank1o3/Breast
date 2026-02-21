#pragma once
// engine/solver.hpp

#include <cstdint>
#include <vector>

namespace breast::engine {

struct PointInit {
    float x, y, z;
    bool  pinned = false;
};

struct SpringInit {
    int32_t a, b;
    float   stiffness   = 1.0f;
    float   rest_length = 0.0f;
};

class Solver {
public:
    // ── Tuneable parameters ──────────────────────────────────────────────────
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

    // ── Position buffers (zero-copy numpy access via bindings) ───────────────
    // Layout: flat float32, length N*3  (x0,y0,z0, x1,y1,z1, ...)
    std::vector<float> pos;
    std::vector<float> prev_pos;

    // ── Construction ─────────────────────────────────────────────────────────
    Solver(
        const std::vector<PointInit>&  points,
        const std::vector<SpringInit>& springs,
        const std::vector<int32_t>&    faces_flat,
        float gravity_y = -9.8f,
        float scale     = 1.0f
    );

    void update(float dt);

    int  num_points()  const;
    int  num_springs() const;
    int  num_faces()   const;

    void set_positions(const float* data, int n);

private:
    std::vector<int32_t> faces_;
    int                  n_faces_ = 0;

    std::vector<bool>    pinned_;
    std::vector<float>   pinned_pos_;

    std::vector<int32_t> spring_a_, spring_b_;
    std::vector<float>   rest_len_, spring_k_;

    std::vector<float>   delta_;   // N*3
    std::vector<float>   counts_;  // N
    std::vector<float>   corr_;    // S*3

    void  verlet_integrate_(float dt, float max_vel);
    void  compute_spring_corrections_(int S);
    void  scatter_spring_corrections_(int S);
    void  apply_deltas_(int N);
    void  apply_pressure_(float pval);
    void  enforce_pinned_();
    float calc_volume_() const;
};

} // namespace breast::engine
