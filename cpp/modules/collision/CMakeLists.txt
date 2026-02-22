# modules/collision/CMakeLists.txt
# Builds:  libcollision.so / collision.dll
#
# Standalone — no dependency on engine or any other module.
# Can be used in entirely separate projects by copying this directory.
# ─────────────────────────────────────────────────────────────────────────────

set(COLLISION_SOURCES
    src/narrow.cpp
    src/world.cpp
    src/broadphase.cpp
)

add_library(collision SHARED ${COLLISION_SOURCES})

target_include_directories(collision
    PUBLIC  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:include>"
)

softsim_compile_options(collision)
softsim_enable_openmp(collision)

# Define COLLISION_OPENMP so the #ifdef guards in the source activate
if(OpenMP_CXX_FOUND)
    target_compile_definitions(collision PRIVATE COLLISION_OPENMP=1)
endif()

set_target_properties(collision PROPERTIES
    VERSION   1.0.0
    SOVERSION 1
)

message(STATUS "[collision] configured — ${COLLISION_SOURCES}")

install(TARGETS collision
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
