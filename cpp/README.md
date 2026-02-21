# cpp/ — C++ extension modules for softsim

## Directory layout

```
cpp/
├── CMakeLists.txt              ← Root: finds Python, pybind11, OpenMP;
│                                 add_subdirectory for each module + bindings
├── cmake/
│   └── common.cmake            ← Shared compile flags + install helpers
│
├── modules/                    ← One subdirectory per C++ shared library
│   └── engine/                 ← Soft-body physics (libengine.so)
│       ├── CMakeLists.txt
│       ├── include/
│       │   └── engine/
│       │       ├── solver.hpp  ← Public API (included by bindings + other modules)
│       │       └── volume.hpp
│       └── src/
│           ├── solver.cpp
│           └── volume.cpp
│
└── bindings/                   ← One .cpp file per Python extension module
    ├── CMakeLists.txt          ← softsim_binding() macro, one call per module
    └── bind_engine.cpp         ← _engine Python module (thin pybind11 shim)
```

## Adding a new module

### 1. Create the shared lib

```
cpp/modules/newmod/
├── CMakeLists.txt
├── include/newmod/api.hpp
└── src/impl.cpp
```

`modules/newmod/CMakeLists.txt` — minimal template:
```cmake
set(NEWMOD_SOURCES src/impl.cpp)
add_library(newmod SHARED ${NEWMOD_SOURCES})
target_include_directories(newmod PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
softsim_compile_options(newmod)
softsim_enable_openmp(newmod)   # if it uses #pragma omp
```

If `newmod` needs types from `engine`, just add:
```cmake
target_link_libraries(newmod PRIVATE engine)
```
No code is copied — `newmod` links `libengine.so` at runtime.

### 2. Register it in the root CMakeLists.txt

```cmake
add_subdirectory(modules/newmod)
```

### 3. Add a binding

Create `bindings/bind_newmod.cpp`, then in `bindings/CMakeLists.txt`:

```cmake
softsim_binding(
    NAME    _newmod
    SOURCES bind_newmod.cpp
    LINKS   newmod engine   # list every shared lib it needs
)
```

### 4. Add to stub generation (root CMakeLists.txt)

```cmake
add_custom_target(stubs ...
    COMMAND ... pybind11_stubgen ... _newmod
    DEPENDS _engine _newmod
)
```

## Build

```bash
pip install pybind11 pybind11-stubgen numpy
./build.sh                    # Release
./build.sh Debug              # Debug + ASan
./build.sh Release stubs      # Release + generate .pyi stubs
```

## Migration (one line in sim.py)

```python
# Before
from breast.solver_numpy import UltraStableSolver
# After
from breast.solver_cpp import UltraStableSolver
```

`solver_cpp.py` falls back to `solver_numpy` with a warning if `_engine`
is not built, so nothing hard-breaks.
