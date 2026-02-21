# bindings/CMakeLists.txt
# ─────────────────────────────────────────────────────────────────────────────
# Each pybind11 extension is declared with breast_binding(name sources... deps...)
# The macro:
#   1. Calls pybind11_add_module()
#   2. Applies shared compile flags
#   3. Links the requested module shared libs
#   4. Copies the .so into ${BREAST_PKG_DIR} after build
# ─────────────────────────────────────────────────────────────────────────────

# ── Helper macro ──────────────────────────────────────────────────────────────
# Usage:
#   breast_binding(
#       NAME    _engine
#       SOURCES bind_engine.cpp
#       LINKS   engine          # shared libs to link
#   )
macro(breast_binding)
    cmake_parse_arguments(_B "" "NAME" "SOURCES;LINKS" ${ARGN})

    pybind11_add_module(${_B_NAME} ${_B_SOURCES})

    # Bindings live in this directory
    target_include_directories(${_B_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    breast_compile_options(${_B_NAME})

    # Link the requested breast module shared libs
    foreach(_lib ${_B_LINKS})
        target_link_libraries(${_B_NAME} PRIVATE ${_lib})
    endforeach()

    # Copy to Python package after build
    breast_install_to_pkg(${_B_NAME})

    message(STATUS "[binding] ${_B_NAME} → links: ${_B_LINKS}")
endmacro()

# ─────────────────────────────────────────────────────────────────────────────
# Bindings — add one breast_binding() call per extension module
# ─────────────────────────────────────────────────────────────────────────────

breast_binding(
    NAME    _engine
    SOURCES bind_engine.cpp
    LINKS   engine
)

# Future bindings:
#
# breast_binding(
#     NAME    _collision
#     SOURCES bind_collision.cpp
#     LINKS   collision engine   # collision depends on engine — both linked here
# )
#
# breast_binding(
#     NAME    _rendering
#     SOURCES bind_rendering.cpp
#     LINKS   rendering
# )
