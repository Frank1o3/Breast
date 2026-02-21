# cmake/common.cmake
# ─────────────────────────────────────────────────────────────────────────────
# Shared compile flags applied to every target via softsim_compile_options().
# Shared libs and pybind11 extension modules both call this.
# ─────────────────────────────────────────────────────────────────────────────

# ── Optimisation / warning flags ─────────────────────────────────────────────
function(softsim_compile_options target)
    target_compile_options(${target} PRIVATE
        # GCC / Clang
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
            -Wall -Wextra -Wpedantic
            $<$<CONFIG:Release>:-O3 -march=native -ffast-math -DNDEBUG>
            $<$<CONFIG:Debug>:  -O0 -g -fsanitize=address,undefined>
        >
        # MSVC
        $<$<CXX_COMPILER_ID:MSVC>:
            /W4
            $<$<CONFIG:Release>:/O2 /fp:fast /arch:AVX2 /DNDEBUG>
            $<$<CONFIG:Debug>:  /Od /Zi /RTC1>
        >
    )

    # ASan link flags for Debug on GCC/Clang
    target_link_options(${target} PRIVATE
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:
            -fsanitize=address,undefined
        >
    )
endfunction()

# ── OpenMP helper ─────────────────────────────────────────────────────────────
# Call softsim_enable_openmp(target) on any target that uses #pragma omp.
function(softsim_enable_openmp target)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${target} PRIVATE SOFTSIM_OPENMP=1)
    endif()
endfunction()

# ── Copy-to-package helper ────────────────────────────────────────────────────
# softsim_install_to_pkg(target)
#   After build, copies the compiled .so / .pyd into ${BREAST_PKG_DIR}.
#   Used by every pybind11 extension target in bindings/CMakeLists.txt.
function(softsim_install_to_pkg target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${target}>"
                "${BREAST_PKG_DIR}/$<TARGET_FILE_NAME:${target}>"
        COMMENT "Installing ${target} → ${BREAST_PKG_DIR}"
    )
endfunction()
