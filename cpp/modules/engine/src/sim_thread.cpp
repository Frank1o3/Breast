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
        // Save a deep copy of the initial state for reset
        initial_state_ = std::make_unique<Solver>(*solver_);

        // Seed atomics from solver defaults so ramp starts from a known state
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
    // ─────────────────────────────────────────────────────────────────────────────

    void SimThread::loop_()
    {
        using clock = std::chrono::steady_clock;
        const float dt = 1.0f / static_cast<float>(physics_fps_);
        const float sub_dt = dt / static_cast<float>(sub_steps_);
        const auto frame = std::chrono::duration<float>(dt);

        float cur_stiffness = solver_->stiffness;
        float cur_pressure = solver_->pressure_stiffness;

        while (running_.load())
        {
            auto t0 = clock::now();

            if (reset_flag_.exchange(false))
            {
                *solver_ = *initial_state_;
                cur_stiffness = solver_->stiffness;
                cur_pressure = solver_->pressure_stiffness;
            }

            ramp_(cur_stiffness, target_stiffness_.load(), 2.5f, dt);
            ramp_(cur_pressure, target_pressure_.load(), 2.5f, dt);

            solver_->stiffness = cur_stiffness;
            solver_->pressure_stiffness = cur_pressure;

            for (int i = 0; i < sub_steps_; ++i)
                solver_->update(sub_dt);

            if (solver_->is_exploded)
            {
                *solver_ = *initial_state_;
                cur_stiffness = solver_->stiffness;
                cur_pressure = solver_->pressure_stiffness;
            }

            auto elapsed = clock::now() - t0;
            if (elapsed < frame)
                std::this_thread::sleep_for(frame - elapsed);
        }
    }

} // namespace breast::engine
