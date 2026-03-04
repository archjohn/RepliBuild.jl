# Verification and Benchmarking script for zero-copy strided array performance

using Test
using Pkg

# Ensure RepliBuild is available
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using RepliBuild

@testset "Zero-Copy Performance Benchmark" begin
    println("\n" * "="^70)
    println("Building and Wrapping Benchmark Test...")
    println("="^70)

    toml_path = joinpath(@__DIR__, "replibuild.toml")
    @test isfile(toml_path)

    # Build and Wrap
    library_path = RepliBuild.build(toml_path)
    @test isfile(library_path)
    
    wrapper_path = RepliBuild.wrap(toml_path)
    @test isfile(wrapper_path)
    
    # Load wrapper
    include(wrapper_path)
    
    println("\nTesting Zero-Copy Strided Matrix View...")

    # Matrix dimensions
    N = 200 # Using a modest size for testing logic first

    # Create standard Julia matrices (Column-major by default)
    # They are contiguous in memory but have a specific stride structure
    A = rand(Float64, N, N)
    B = rand(Float64, N, N)
    C_jl = zeros(Float64, N, N)
    C_cpp = zeros(Float64, N, N)

    # Calculate exact strided views mapping for Julia matrices (Column-major)
    # A Julia Matrix (2D Array) has:
    # stride_row = 1 (elements in a column are contiguous)
    # stride_col = size(Matrix, 1) (distance between columns is the number of rows)
    
    viewA = BenchmarkTest.StridedMatrixView(
        pointer(A), 
        UInt64(N), UInt64(N), 
        UInt64(1), UInt64(N)
    )
    
    viewB = BenchmarkTest.StridedMatrixView(
        pointer(B), 
        UInt64(N), UInt64(N), 
        UInt64(1), UInt64(N)
    )
    
    viewC = BenchmarkTest.StridedMatrixView(
        pointer(C_cpp), 
        UInt64(N), UInt64(N), 
        UInt64(1), UInt64(N)
    )

    # Base native ccall verification
    println("Executing C++ strided matrix multiplication...")
    
    # Ensure memory is pinned during ccall using Ref wrappers
    GC.@preserve A B C_cpp begin
        ref_viewA = Ref(viewA)
        ref_viewB = Ref(viewB)
        ref_viewC = Ref(viewC)
        
        # Invoke wrapped C++ function natively using pointer to structs
        BenchmarkTest.multiply_matrices(
            Base.unsafe_convert(Ptr{BenchmarkTest.StridedMatrixView}, ref_viewA),
            Base.unsafe_convert(Ptr{BenchmarkTest.StridedMatrixView}, ref_viewB),
            Base.unsafe_convert(Ptr{BenchmarkTest.StridedMatrixView}, ref_viewC)
        )
    end
    
    # Compute native Julia multiplication for validation
    println("Executing Julia matrix multiplication...")
    C_jl = A * B

    # Validate computational correctness 
    # (checking if C++ correctly computed strided pointers across matrices)
    @test C_cpp ≈ C_jl
    println("✓ Mathematical validation passed: Julia and C++ computed the identical result.")
end

println("\nRunning Performance Benchmarks...")
println("Note: Using Pkg 'BenchmarkTools' dynamically to test raw throughput.")

# Add BenchmarkTools temporarily
Pkg.add("BenchmarkTools")
using BenchmarkTools

include(joinpath(@__DIR__, "julia", "BenchmarkTest.jl"))
using .BenchmarkTest
using Libdl

# Re-declare variables for benchmark scope
N = 200
A = rand(Float64, N, N)
B = rand(Float64, N, N)
C_cpp = zeros(Float64, N, N)

viewA = BenchmarkTest.StridedMatrixView(pointer(A), UInt64(N), UInt64(N), UInt64(1), UInt64(N))
viewB = BenchmarkTest.StridedMatrixView(pointer(B), UInt64(N), UInt64(N), UInt64(1), UInt64(N))
viewC = BenchmarkTest.StridedMatrixView(pointer(C_cpp), UInt64(N), UInt64(N), UInt64(1), UInt64(N))

println("\n1. Julia Native `A * B`")
b_jl = @benchmark $A * $B
display(b_jl)

println("\n\n2. RepliBuild Wrapper `multiply_matrices` (Zero-copy Struct Pointer)")
b_cpp = @benchmark GC.@preserve $A $B $C_cpp begin
    ref_viewA = Ref($viewA)
    ref_viewB = Ref($viewB)
    ref_viewC = Ref($viewC)
    BenchmarkTest.multiply_matrices(
        Base.unsafe_convert(Ptr{BenchmarkTest.StridedMatrixView}, ref_viewA),
        Base.unsafe_convert(Ptr{BenchmarkTest.StridedMatrixView}, ref_viewB),
        Base.unsafe_convert(Ptr{BenchmarkTest.StridedMatrixView}, ref_viewC)
    )
end
display(b_cpp)

println("\n\n3. Bare-metal ccall `multiply_matrices`")
library_path = joinpath(@__DIR__, "julia", "libbenchmark_test.so")
lib_sym = Libdl.dlsym(Libdl.dlopen(library_path), :multiply_matrices)
b_bare = @benchmark GC.@preserve $A $B $C_cpp begin
    ref_viewA = Ref($viewA)
    ref_viewB = Ref($viewB)
    ref_viewC = Ref($viewC)
    ccall($lib_sym, Cvoid, 
        (Ptr{BenchmarkTest.StridedMatrixView}, Ptr{BenchmarkTest.StridedMatrixView}, Ptr{BenchmarkTest.StridedMatrixView}),
        Base.unsafe_convert(Ptr{BenchmarkTest.StridedMatrixView}, ref_viewA),
        Base.unsafe_convert(Ptr{BenchmarkTest.StridedMatrixView}, ref_viewB),
        Base.unsafe_convert(Ptr{BenchmarkTest.StridedMatrixView}, ref_viewC))
end
display(b_bare)
println()
