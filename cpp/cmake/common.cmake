# cmake/common.cmake
# ─────────────────────────────────────────────────────────────────────────────
# Shared compile flags applied to every target via softsim_compile_options().
# Shared libs and pybind11 extension modules both call this.
# ─────────────────────────────────────────────────────────────────────────────

# ── Optimisation / warning flags ─────────────────────────────────────────────
function(softsim_compile_options target)

    # ─────────────────────────────────────────────────────────────────────────
    # GCC / Clang
    # ─────────────────────────────────────────────────────────────────────────
    if(NOT MSVC)

        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -fPIC
        )
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE O)

        if(CMAKE_BUILD_TYPE STREQUAL "Release")

            if(SOFTSIM_PACKAGING)
                target_compile_options(${target} PRIVATE
                    -O3 -ffast-math -DNDEBUG
                )
            else()
                target_compile_options(${target} PRIVATE
                    -O3 -march=native -ffast-math -DNDEBUG
                )
            endif()

        elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")

            target_compile_options(${target} PRIVATE
                -O0 -g -fsanitize=address,undefined
            )

            target_link_options(${target} PRIVATE
                -fsanitize=address,undefined
            )

        endif()

    # ─────────────────────────────────────────────────────────────────────────
    # MSVC
    # ─────────────────────────────────────────────────────────────────────────
    else()

        target_compile_options(${target} PRIVATE /W4)

        if(CMAKE_BUILD_TYPE STREQUAL "Release")

            if(SOFTSIM_PACKAGING)
                target_compile_options(${target} PRIVATE
                    /O2 /fp:fast /DNDEBUG
                )
            else()
                target_compile_options(${target} PRIVATE
                    /O2 /fp:fast /arch:AVX2 /DNDEBUG
                )
            endif()

        elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")

            target_compile_options(${target} PRIVATE
                /Od /Zi /RTC1
            )

        endif()

    endif()

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
#   If BREAST_CORE_ONLY is OFF, also generates __init__.py.
#   Used by every pybind11 extension target in bindings/CMakeLists.txt.
function(softsim_install_to_pkg target)

    # Derive subpackage name from target (_engine → engine)
    string(REGEX REPLACE "^_" "" _subdir "${target}")

    # ─────────────────────────────────────────────────────────────────────────
    # Packaging mode (scikit-build-core)
    # ─────────────────────────────────────────────────────────────────────────
    if(SKBUILD)

        # Always install the extension module
        install(
            TARGETS ${target}
            LIBRARY DESTINATION breast/${_subdir}
            RUNTIME DESTINATION breast/${_subdir}
        )

        # Only install Python package files if NOT core-only
        if(NOT BREAST_CORE_ONLY)
            install(
                FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
                DESTINATION breast/${_subdir}
                OPTIONAL
            )
        endif()

    # ─────────────────────────────────────────────────────────────────────────
    # Manual dev mode (build.sh)
    # ─────────────────────────────────────────────────────────────────────────
    elseif(BREAST_PKG_DIR)

        set(_dest "${BREAST_PKG_DIR}/${_subdir}")

        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_dest}"

            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:${target}>"
                    "${_dest}/$<TARGET_FILE_NAME:${target}>"
        )

        # Only generate __init__.py if NOT core-only
        if(NOT BREAST_CORE_ONLY)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                        -D DEST_FILE="${_dest}/__init__.py"
                        -D MODULE="${target}"
                        -P "${CMAKE_SOURCE_DIR}/cmake/write_init.cmake"
            )
        endif()

        add_custom_command(TARGET ${target} POST_BUILD
            COMMENT "Installing ${target} → ${_dest}"
        )

    endif()
endfunction()
