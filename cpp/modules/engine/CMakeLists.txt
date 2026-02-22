# modules/engine/CMakeLists.txt
# Builds:  libengine.so / engine.dll
#
# Other modules that need engine types/functions link against this target:
#   target_link_libraries(libcollision PRIVATE engine)
#
# The pybind11 binding (bindings/bind_engine.cpp) also links this target.
# ─────────────────────────────────────────────────────────────────────────────

# ── Source files ──────────────────────────────────────────────────────────────
# Add more .cpp files here as the engine grows.
set(ENGINE_SOURCES
    src/solver.cpp
    src/volume.cpp
    src/sim_thread.cpp
)

# ── Shared library ────────────────────────────────────────────────────────────
add_library(engine SHARED ${ENGINE_SOURCES})

# Public include — anything that links engine gets engine/ on its include path.
# This means clients write:  #include "engine/solver.hpp"
target_include_directories(engine
    PUBLIC  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:include>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../collision/include"
)

# Apply shared compile flags
softsim_compile_options(engine)

# OpenMP — engine uses #pragma omp in hot loops
softsim_enable_openmp(engine)

# ── Dependencies ──────────────────────────────────────────────────────────────
# Engine depends on collision for world/constraint handling
target_link_libraries(engine PRIVATE collision)

# ── Module-local dependencies ─────────────────────────────────────────────────
# Add find_package() calls here for deps that ONLY engine needs.
# Example:
#   find_package(Eigen3 REQUIRED)
#   target_link_libraries(engine PRIVATE Eigen3::Eigen)

# ── Versioning (optional but good practice) ───────────────────────────────────
set_target_properties(engine PROPERTIES
    VERSION   1.0.0
    SOVERSION 1
)

message(STATUS "[engine] configured — ${ENGINE_SOURCES}")
