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
    # Derive a clean subdir name by stripping the leading underscore.
    # _engine  →  engine,   _collision  →  collision, etc.
    string(REGEX REPLACE "^_" "" _subdir "${target}")
    set(_dest "${BREAST_PKG_DIR}/${_subdir}")

    add_custom_command(TARGET ${target} POST_BUILD
        # 1. Make the subpackage directory
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_dest}"

        # 2. Copy the compiled extension (.so / .pyd)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${target}>"
                "${_dest}/$<TARGET_FILE_NAME:${target}>"

        # 3. Generate __init__.py only if it does not already exist.
        #    file(GENERATE ...) runs at generate-time, not build-time,
        #    so we use a small cmake -P script instead to get the
        #    "don't overwrite" behaviour.
        COMMAND ${CMAKE_COMMAND}
                -D DEST_FILE="${_dest}/__init__.py"
                -D MODULE="${target}"
                -P "${CMAKE_SOURCE_DIR}/cmake/write_init.cmake"

        COMMENT "Installing ${target} → ${_dest}"
    )
endfunction()
