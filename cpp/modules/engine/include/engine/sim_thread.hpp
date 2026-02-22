#pragma once
// engine/sim_thread.hpp

#include "engine/solver.hpp"
#include "collision/world.hpp"

#include <atomic>
#include <memory>
#include <thread>

namespace breast::engine
{

    class SimThread
    {
    public:
        SimThread(
            std::unique_ptr<Solver> solver,
            int physics_fps = 120,
            int sub_steps = 5);
        ~SimThread();

        // ── Thread control ────────────────────────────────────────────────────────
        void start();
        void stop();
        bool is_running() const { return running_.load(); }

        // ── Parameter control (lock-free, ramped inside loop) ─────────────────────
        void set_stiffness(float v) { target_stiffness_.store(v); }
        void set_pressure(float v) { target_pressure_.store(v); }
        void reset() { reset_flag_.store(true); }

        // ── Position access (zero-copy view into solver buffer) ───────────────────
        const float *pos_data() const { return solver_->pos.data(); }
        int num_points() const { return solver_->num_points(); }

        // ── Collision world — owned by SimThread, configured from Python/sim.py ───
        // The loop calls world.resolve() each tick before spring constraints.
        // Add colliders via sim_thread.collision_world.add_sphere(...) etc.
        collision::World collision_world;

    private:
        std::unique_ptr<Solver> solver_;
        std::unique_ptr<Solver> initial_state_;

        std::thread thread_;
        std::atomic<bool> running_{false};
        std::atomic<bool> reset_flag_{false};

        std::atomic<float> target_stiffness_;
        std::atomic<float> target_pressure_;

        int physics_fps_;
        int sub_steps_;

        void loop_();
        void ramp_(float &current, float target, float rate, float dt);
    };

} // namespace breast::engine
