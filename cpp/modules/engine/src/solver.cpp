// engine/src/solver.cpp
#include "engine/solver.hpp"
#include "engine/volume.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#ifdef breast_OPENMP
#include <omp.h>
#endif

namespace breast::engine {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────
Solver::Solver(
    const std::vector<PointInit>&  points,
    const std::vector<SpringInit>& springs,
    const std::vector<int32_t>&    faces_flat,
    float grav_y,
    float scale
) {
    const int N = static_cast<int>(points.size());
    const int S = static_cast<int>(springs.size());
    const int F = static_cast<int>(faces_flat.size()) / 3;

    gravity_y = grav_y;
    faces_    = faces_flat;
    n_faces_  = F;

    // ── Positions ────────────────────────────────────────────────────────────
    pos.resize(N * 3);
    prev_pos.resize(N * 3);
    pinned_.resize(N, false);
    pinned_pos_.resize(N * 3, 0.f);

    for (int i = 0; i < N; ++i) {
        pos[i*3]   = points[i].x * scale;
        pos[i*3+1] = points[i].y * scale;
        pos[i*3+2] = points[i].z * scale;
        pinned_[i] = points[i].pinned;
    }
    prev_pos = pos; // zero initial velocity

    for (int i = 0; i < N; ++i) {
        if (!pinned_[i]) continue;
        pinned_pos_[i*3]   = pos[i*3];
        pinned_pos_[i*3+1] = pos[i*3+1];
        pinned_pos_[i*3+2] = pos[i*3+2];
    }

    // ── Springs ──────────────────────────────────────────────────────────────
    spring_a_.resize(S);
    spring_b_.resize(S);
    rest_len_.resize(S);
    spring_k_.resize(S);

    double edge_sum = 0.0;
    for (int s = 0; s < S; ++s) {
        spring_a_[s] = springs[s].a;
        spring_b_[s] = springs[s].b;
        rest_len_[s] = springs[s].rest_length * scale;
        spring_k_[s] = springs[s].stiffness;
        edge_sum    += rest_len_[s];
    }
    avg_edge = S > 0 ? static_cast<float>(edge_sum / S) : 0.f;

    // ── Work buffers ─────────────────────────────────────────────────────────
    delta_.resize(N * 3, 0.f);
    counts_.resize(N,     0.f);
    corr_.resize(S * 3,   0.f);

    // ── Rest volume ──────────────────────────────────────────────────────────
    rest_volume = calc_volume_();
}

// ─────────────────────────────────────────────────────────────────────────────
// Accessors
// ─────────────────────────────────────────────────────────────────────────────
int Solver::num_points()  const { return static_cast<int>(pinned_.size()); }
int Solver::num_springs() const { return static_cast<int>(spring_a_.size()); }
int Solver::num_faces()   const { return n_faces_; }

void Solver::set_positions(const float* data, int n) {
    if (n != static_cast<int>(pos.size()))
        throw std::runtime_error("set_positions: size mismatch");
    std::copy(data, data + n, pos.begin());
}

// ─────────────────────────────────────────────────────────────────────────────
// Main update
// ─────────────────────────────────────────────────────────────────────────────
void Solver::update(float dt) {
    if (is_exploded) return;

    const int   N       = num_points();
    const int   S       = num_springs();
    const float max_vel = avg_edge * 0.5f / dt;

    // 1. Verlet integration
    verlet_integrate_(dt, max_vel);

    // 2. Jacobi spring iterations
    for (int iter = 0; iter < iterations; ++iter) {
        std::fill(delta_.begin(),  delta_.end(),  0.f);
        std::fill(counts_.begin(), counts_.end(), 0.f);
        compute_spring_corrections_(S);
        scatter_spring_corrections_(S);
        apply_deltas_(N);
    }

    // 3. Pressure
    const float cur_vol = calc_volume_();
    const float denom   = std::max(rest_volume, 1e-8f);
    const float vol_err = std::max(-0.3f, std::min(0.3f,
                              (rest_volume - cur_vol) / denom));
    const float pval    = vol_err * pressure_stiffness;
    if (std::abs(pval) > 1e-12f)
        apply_pressure_(pval);

    // 4. Pin enforcement
    enforce_pinned_();

    // 5. Stability check
    for (float v : pos) {
        if (!std::isfinite(v)) {
            is_exploded = true;
            return;
        }
    }
    ++steps_stable;
}

// ─────────────────────────────────────────────────────────────────────────────
// Verlet integration
// ─────────────────────────────────────────────────────────────────────────────
void Solver::verlet_integrate_(float dt, float max_vel) {
    const int   N   = num_points();
    const float dt2 = dt * dt;
    const float gy  = gravity_y;
    const float mv2 = max_vel * max_vel;

    #pragma omp parallel for schedule(static) if(N > 512)
    for (int i = 0; i < N; ++i) {
        if (pinned_[i]) continue;

        const int b = i * 3;
        float vx = (pos[b]   - prev_pos[b])   * damping;
        float vy = (pos[b+1] - prev_pos[b+1]) * damping;
        float vz = (pos[b+2] - prev_pos[b+2]) * damping;

        // Velocity clamp
        const float spd2 = vx*vx + vy*vy + vz*vz;
        if (spd2 > mv2) {
            const float inv = max_vel / std::sqrt(spd2);
            vx *= inv; vy *= inv; vz *= inv;
        }

        prev_pos[b]   = pos[b];
        prev_pos[b+1] = pos[b+1];
        prev_pos[b+2] = pos[b+2];

        pos[b]   += vx;
        pos[b+1] += vy + gy * dt2;
        pos[b+2] += vz;

        // Floor
        if (pos[b+1] < floor_y) {
            pos[b+1]      = floor_y;
            prev_pos[b+1] = floor_y;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spring correction — parallel compute phase
// ─────────────────────────────────────────────────────────────────────────────
void Solver::compute_spring_corrections_(int S) {
    #pragma omp parallel for schedule(static) if(S > 512)
    for (int s = 0; s < S; ++s) {
        const int a  = spring_a_[s];
        const int b  = spring_b_[s];
        const int ab = a * 3, bb = b * 3;

        const float dx = pos[bb]   - pos[ab];
        const float dy = pos[bb+1] - pos[ab+1];
        const float dz = pos[bb+2] - pos[ab+2];
        const float d2 = dx*dx + dy*dy + dz*dz;

        if (d2 < 1e-10f) {
            corr_[s*3] = corr_[s*3+1] = corr_[s*3+2] = 0.f;
            continue;
        }

        const float dist = std::sqrt(d2);
        // Effective stiffness = per-spring k * global multiplier
        const float k = spring_k_[s] * stiffness * 0.5f
                        * (1.f - rest_len_[s] / dist);

        corr_[s*3]   = dx * k;
        corr_[s*3+1] = dy * k;
        corr_[s*3+2] = dz * k;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spring correction — sequential scatter (avoids write races)
// ─────────────────────────────────────────────────────────────────────────────
void Solver::scatter_spring_corrections_(int S) {
    for (int s = 0; s < S; ++s) {
        const int   a  = spring_a_[s];
        const int   b  = spring_b_[s];
        const float cx = corr_[s*3];
        const float cy = corr_[s*3+1];
        const float cz = corr_[s*3+2];

        if (!pinned_[a]) {
            delta_[a*3]   += cx;
            delta_[a*3+1] += cy;
            delta_[a*3+2] += cz;
            counts_[a]    += 1.f;
        }
        if (!pinned_[b]) {
            delta_[b*3]   -= cx;
            delta_[b*3+1] -= cy;
            delta_[b*3+2] -= cz;
            counts_[b]    += 1.f;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Apply averaged Jacobi deltas
// ─────────────────────────────────────────────────────────────────────────────
void Solver::apply_deltas_(int N) {
    #pragma omp parallel for schedule(static) if(N > 512)
    for (int i = 0; i < N; ++i) {
        if (pinned_[i] || counts_[i] < 0.5f) continue;
        const float inv = 1.f / counts_[i];
        pos[i*3]   += delta_[i*3]   * inv;
        pos[i*3+1] += delta_[i*3+1] * inv;
        pos[i*3+2] += delta_[i*3+2] * inv;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pressure — outward face-normal forces
// ─────────────────────────────────────────────────────────────────────────────
void Solver::apply_pressure_(float pval) {
    const int F = n_faces_;
    // Temp face-force vectors (stack-allocated for small meshes, heap for large)
    std::vector<float> fvec(static_cast<size_t>(F) * 3);

    #pragma omp parallel for schedule(static) if(F > 256)
    for (int i = 0; i < F; ++i) {
        const int i0 = faces_[i*3], i1 = faces_[i*3+1], i2 = faces_[i*3+2];

        const float e0x = pos[i1*3]   - pos[i0*3];
        const float e0y = pos[i1*3+1] - pos[i0*3+1];
        const float e0z = pos[i1*3+2] - pos[i0*3+2];
        const float e1x = pos[i2*3]   - pos[i0*3];
        const float e1y = pos[i2*3+1] - pos[i0*3+1];
        const float e1z = pos[i2*3+2] - pos[i0*3+2];

        const float nx = e0y*e1z - e0z*e1y;
        const float ny = e0z*e1x - e0x*e1z;
        const float nz = e0x*e1y - e0y*e1x;
        const float s  = pval / 3.f;

        fvec[i*3]   = nx * s;
        fvec[i*3+1] = ny * s;
        fvec[i*3+2] = nz * s;
    }

    // Sequential scatter
    for (int i = 0; i < F; ++i) {
        const int   i0 = faces_[i*3], i1 = faces_[i*3+1], i2 = faces_[i*3+2];
        const float fx = fvec[i*3], fy = fvec[i*3+1], fz = fvec[i*3+2];

        if (!pinned_[i0]) { pos[i0*3]+=fx; pos[i0*3+1]+=fy; pos[i0*3+2]+=fz; }
        if (!pinned_[i1]) { pos[i1*3]+=fx; pos[i1*3+1]+=fy; pos[i1*3+2]+=fz; }
        if (!pinned_[i2]) { pos[i2*3]+=fx; pos[i2*3+1]+=fy; pos[i2*3+2]+=fz; }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-enforce pinned vertices
// ─────────────────────────────────────────────────────────────────────────────
void Solver::enforce_pinned_() {
    const int N = num_points();
    for (int i = 0; i < N; ++i) {
        if (!pinned_[i]) continue;
        pos[i*3]   = pinned_pos_[i*3];
        pos[i*3+1] = pinned_pos_[i*3+1];
        pos[i*3+2] = pinned_pos_[i*3+2];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Volume  (delegates to volume.cpp)
// ─────────────────────────────────────────────────────────────────────────────
float Solver::calc_volume_() const {
    return calc_mesh_volume(pos, faces_);
}

} // namespace breast::engine
