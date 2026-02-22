// engine/include/engine/sim_thread.hpp
#pragma once
#include "engine/solver.hpp"
#include <atomic>
#include <thread>
#include <mutex>

namespace breast::engine
{

    class SimThread
    {
    public:
        explicit SimThread(std::unique_ptr<Solver> solver,
                           int physics_fps = 120,
                           int sub_steps = 5);
        ~SimThread();

        // Zero-copy access — caller must hold lock or accept tearing
        // (tearing is acceptable for rendering — a slightly stale frame is fine)
        const float *pos_data() const { return solver_->pos.data(); }
        int num_points() const { return solver_->num_points(); }

        // Thread control
        void start();
        void stop();

        // Parameter updates (lock-free atomics — no stall on render thread)
        void set_stiffness(float v) { target_stiffness_.store(v); }
        void set_pressure(float v) { target_pressure_.store(v); }
        void reset() { reset_flag_.store(true); }

        bool is_running() const { return running_.load(); }

    private:
        void loop_();
        void ramp_(float &current, float target, float rate, float dt);

        std::unique_ptr<Solver> solver_;
        std::unique_ptr<Solver> initial_state_; // saved for reset

        std::thread thread_;
        std::atomic<bool> running_{false};
        std::atomic<bool> reset_flag_{false};

        std::atomic<float> target_stiffness_;
        std::atomic<float> target_pressure_;

        int physics_fps_;
        int sub_steps_;
    };

} // namespace breast::engine
