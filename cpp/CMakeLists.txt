cmake_minimum_required(VERSION 3.22)
project(softsim_ext LANGUAGES CXX)

# ─────────────────────────────────────────────────────────────────────────────
# C++ standard (all sub-projects inherit this)
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
# Output directories — all .so/.pyd land in build/lib
# ─────────────────────────────────────────────────────────────────────────────
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ─────────────────────────────────────────────────────────────────────────────
# Where to copy finished extension modules into the Python package.
# build.sh passes -DBREAST_PKG_DIR automatically.
# ─────────────────────────────────────────────────────────────────────────────
set(BREAST_PKG_DIR "${CMAKE_SOURCE_DIR}/../src/breast"
    CACHE PATH "Destination for built Python extension modules")

# ─────────────────────────────────────────────────────────────────────────────
# Shared CMake helpers (compile flags, install macros)
# ─────────────────────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(common)

# ─────────────────────────────────────────────────────────────────────────────
# Python
# build.sh always passes -DPython_EXECUTABLE so this picks up the venv Python.
# ─────────────────────────────────────────────────────────────────────────────
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python: ${Python_EXECUTABLE} (${Python_VERSION})")

# ─────────────────────────────────────────────────────────────────────────────
# pybind11 — four-level resolution, first hit wins:
#
#  1. System cmake config    (apt install pybind11-dev)
#  2. Active Python package  (pip/poetry/uv install pybind11)
#  3. Git submodule          (third_party/pybind11  — offline-safe)
#  4. FetchContent download  (network, last resort)
# ─────────────────────────────────────────────────────────────────────────────

# Level 1 — system cmake package
find_package(pybind11 CONFIG QUIET)

# Level 2 — ask the resolved Python (covers pip/poetry/uv envs)
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

# Level 3 — git submodule (third_party/pybind11)
if(NOT pybind11_FOUND)
    set(_pb11_submodule "${CMAKE_SOURCE_DIR}/third_party/pybind11")
    if(EXISTS "${_pb11_submodule}/CMakeLists.txt")
        message(STATUS "pybind11: using git submodule at ${_pb11_submodule}")
        add_subdirectory("${_pb11_submodule}" pybind11_sub EXCLUDE_FROM_ALL)
        set(pybind11_FOUND TRUE)
    endif()
endif()

# Level 4 — FetchContent (requires network, only triggers if submodule absent)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11: not found locally — fetching via FetchContent (requires network)")
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
    message(FATAL_ERROR
        "pybind11 could not be found by any method.\n"
        "Options:\n"
        "  pip install pybind11                    (pip / venv)\n"
        "  poetry add --group dev pybind11         (poetry)\n"
        "  uv add --dev pybind11                   (uv)\n"
        "  git submodule add https://github.com/pybind/pybind11 cpp/third_party/pybind11\n"
    )
endif()

message(STATUS "pybind11: ${pybind11_VERSION}")

# ─────────────────────────────────────────────────────────────────────────────
# Optional: OpenMP (shared across all modules that opt in)
# ─────────────────────────────────────────────────────────────────────────────
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: ${OpenMP_CXX_VERSION} (${OpenMP_CXX_FLAGS})")
else()
    message(WARNING "OpenMP not found — parallel kernels will run single-threaded")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Modules (each builds a shared lib)
# ─────────────────────────────────────────────────────────────────────────────
add_subdirectory(modules/engine)
# add_subdirectory(modules/collision)
# add_subdirectory(modules/rendering)

# ─────────────────────────────────────────────────────────────────────────────
# Bindings (each builds a pybind11 extension module)
# ─────────────────────────────────────────────────────────────────────────────
add_subdirectory(bindings)

# ─────────────────────────────────────────────────────────────────────────────
# Optional stub generation (cmake --build build --target stubs)
# Requires:  pip install pybind11-stubgen
# ─────────────────────────────────────────────────────────────────────────────
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
