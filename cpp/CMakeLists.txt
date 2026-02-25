cmake_minimum_required(VERSION 3.22)
project(softsim_ext LANGUAGES CXX)

if(NOT DEFINED SKBUILD)
    set(SKBUILD OFF)
endif()

# Detect packaging mode (scikit-build-core defines SKBUILD)
if(DEFINED SKBUILD AND SKBUILD)
    set(SOFTSIM_PACKAGING ON)
else()
    set(SOFTSIM_PACKAGING OFF)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# C++ standard
# ─────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─────────────────────────────────────────────────────────────────────────────
# Build type
# ─────────────────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ─────────────────────────────────────────────────────────────────────────────
# Output directories (manual builds only)
# scikit-build-core ignores these and uses install()
# ─────────────────────────────────────────────────────────────────────────────
if(NOT DEFINED SKBUILD AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Optional local-dev copy destination
# Only used when building manually via build.sh
# scikit-build-core does NOT pass this variable
# ─────────────────────────────────────────────────────────────────────────────
set(BREAST_PKG_DIR ""
    CACHE PATH "Destination for built Python extension modules (dev only)")

if(BREAST_PKG_DIR)
    message(STATUS "Dev mode: extension modules will be copied to ${BREAST_PKG_DIR}")
else()
    message(STATUS "Packaging mode: using install() rules (scikit-build)")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Shared CMake helpers
# ─────────────────────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(common)

# ─────────────────────────────────────────────────────────────────────────────
# Python
# Let build backend control version
# ─────────────────────────────────────────────────────────────────────────────
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python: ${Python_EXECUTABLE} (${Python_VERSION})")

# ─────────────────────────────────────────────────────────────────────────────
# pybind11 resolution
# ─────────────────────────────────────────────────────────────────────────────

find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pb11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pb11_python_result
    )
    if(_pb11_python_result EQUAL 0 AND EXISTS "${_pb11_cmake_dir}")
        list(APPEND CMAKE_PREFIX_PATH "${_pb11_cmake_dir}")
        find_package(pybind11 CONFIG QUIET)
    endif()
endif()

if(NOT pybind11_FOUND)
    set(_pb11_submodule "${CMAKE_SOURCE_DIR}/third_party/pybind11")
    if(EXISTS "${_pb11_submodule}/CMakeLists.txt")
        message(STATUS "pybind11: using git submodule")
        add_subdirectory("${_pb11_submodule}" pybind11_sub EXCLUDE_FROM_ALL)
        set(pybind11_FOUND TRUE)
    endif()
endif()

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v3.0.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
    set(pybind11_FOUND TRUE)
endif()

if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 could not be resolved.")
endif()

message(STATUS "pybind11: ${pybind11_VERSION}")

# ─────────────────────────────────────────────────────────────────────────────
# Optional OpenMP
# ─────────────────────────────────────────────────────────────────────────────
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: ${OpenMP_CXX_VERSION}")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Modules
# ─────────────────────────────────────────────────────────────────────────────
add_subdirectory(modules/collision)
add_subdirectory(modules/engine)

# ─────────────────────────────────────────────────────────────────────────────
# Bindings
# ─────────────────────────────────────────────────────────────────────────────
add_subdirectory(bindings)

# ─────────────────────────────────────────────────────────────────────────────
# Optional stub generation
# Only valid in dev mode (needs local package path)
# ─────────────────────────────────────────────────────────────────────────────
if(BREAST_PKG_DIR)
    add_custom_target(stubs
        COMMENT "Generating Python stubs via pybind11-stubgen"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_BINARY_DIR}/lib:$ENV{PYTHONPATH}"
                "${Python_EXECUTABLE}" -m pybind11_stubgen
                    --output-dir "${BREAST_PKG_DIR}/engine"
                    _engine
        DEPENDS _engine
    )
endif()
