// engine/src/sim_thread.cpp
#include "engine/sim_thread.hpp"

#include <chrono>
#include <cmath>
#include <thread>

namespace breast::engine
{

    // ─────────────────────────────────────────────────────────────────────────────
    // Construction / Destruction
    // ─────────────────────────────────────────────────────────────────────────────

    SimThread::SimThread(
        std::unique_ptr<Solver> solver,
        int physics_fps,
        int sub_steps)
        : solver_(std::move(solver)), physics_fps_(physics_fps), sub_steps_(sub_steps)
    {
        initial_state_ = std::make_unique<Solver>(*solver_);

        target_stiffness_.store(solver_->stiffness);
        target_pressure_.store(solver_->pressure_stiffness);
    }

    SimThread::~SimThread()
    {
        stop();
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Thread control
    // ─────────────────────────────────────────────────────────────────────────────

    void SimThread::start()
    {
        if (running_.load())
            return;
        running_.store(true);
        thread_ = std::thread(&SimThread::loop_, this);
    }

    void SimThread::stop()
    {
        if (!running_.load())
            return;
        running_.store(false);
        if (thread_.joinable())
            thread_.join();
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Ramp helper
    // ─────────────────────────────────────────────────────────────────────────────

    void SimThread::ramp_(float &current, float target, float rate, float dt)
    {
        const float delta = target - current;
        const float step = rate * dt;
        if (std::abs(delta) <= step)
            current = target;
        else
            current += std::copysign(step, delta);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Physics loop
    //
    // Per-tick order:
    //   1. Reset / parameter ramp
    //   2. For each sub-step:
    //        a. Verlet integration        — moves vertices under gravity/velocity
    //        b. Collision resolve         — pushes vertices out of colliders
    //        c. Spring + pressure solve   — restores soft-body shape
    //   3. Explosion guard
    //   4. Rate-limit sleep
    //
    // Collision runs AFTER Verlet so penetrations caused by the integration step
    // are corrected before springs try to pull neighbours into the collider.
    // Running it BEFORE springs means the spring solve works on valid positions,
    // which converges faster and is more stable than the reverse order.
    // ─────────────────────────────────────────────────────────────────────────────

    void SimThread::loop_()
    {
        using clock = std::chrono::steady_clock;
        const float dt = 1.0f / static_cast<float>(physics_fps_);
        const float sub_dt = dt / static_cast<float>(sub_steps_);
        const auto frame = std::chrono::duration<float>(dt);

        float cur_stiffness = solver_->stiffness;
        float cur_pressure = solver_->pressure_stiffness;

        const bool *pinned = solver_->pinned_data();
        const int N = solver_->num_points();

        while (running_.load())
        {
            auto t0 = clock::now();

            // ── Reset ─────────────────────────────────────────────────────────────
            if (reset_flag_.exchange(false))
            {
                *solver_ = *initial_state_;
                cur_stiffness = solver_->stiffness;
                cur_pressure = solver_->pressure_stiffness;
                // pinned pointer stays valid — same layout after copy-assign
            }

            // ── Parameter ramp ────────────────────────────────────────────────────
            ramp_(cur_stiffness, target_stiffness_.load(), 2.5f, dt);
            ramp_(cur_pressure, target_pressure_.load(), 2.5f, dt);

            solver_->stiffness = cur_stiffness;
            solver_->pressure_stiffness = cur_pressure;

            // ── Sub-steps ─────────────────────────────────────────────────────────
            for (int i = 0; i < sub_steps_; ++i)
            {
                // 1. Verlet — integrate velocity + gravity into new positions
                solver_->integrate(sub_dt);

                // 2. Collision — push vertices out of all registered colliders.
                //    Only runs if at least one collider exists (cheap early-out).
                if (collision_world.num_planes() > 0 ||
                    collision_world.num_spheres() > 0 ||
                    collision_world.num_boxes() > 0 ||
                    collision_world.num_capsules() > 0)
                {
                    collision_world.resolve(
                        solver_->pos.data(),
                        pinned,
                        N,
                        0.1f // friction
                    );
                }

                // 3. Springs + pressure + pin enforcement
                solver_->solve_constraints();
            }

            // ── Explosion guard ───────────────────────────────────────────────────
            if (solver_->is_exploded)
            {
                *solver_ = *initial_state_;
                cur_stiffness = solver_->stiffness;
                cur_pressure = solver_->pressure_stiffness;
            }

            // ── Rate-limit ────────────────────────────────────────────────────────
            const auto elapsed = clock::now() - t0;
            if (elapsed < frame)
                std::this_thread::sleep_for(frame - elapsed);
        }
    }

} // namespace breast::engine
