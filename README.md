# RepliBuild.jl

RepliBuild.jl is designed to seamlessly bridge the gap between C++ projects and Julia. It handles the entire lifecycle: discovering source files, compiling them with dependency-aware caching, linking optimized libraries, and automatically generating high-performance Julia wrappers.

## Key Features

- **Dependency-Aware Compilation**: Smart caching ensures only modified files are recompiled.
- **Parallel Builds**: Leverages multi-threading to speed up compilation of large projects.
- **Automatic Wrapping**: Generates `ccall` bindings, struct definitions, and enums directly from binary metadata (DWARF).
- **Introspection Toolkit**: Built-in tools to analyze binary symbols, debug info, optimization passes, and performance.
- **MLIR Integration**: Low-level bindings to MLIR for advanced IR manipulation and JIT compilation.