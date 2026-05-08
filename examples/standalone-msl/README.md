# `standalone-msl` — minimal C++ example using `dist/softfloat64.metal`

Demonstrates that `dist/softfloat64.metal` can be dropped into a plain
Apple Metal C++ project with no Rust dependency.

## Build (manual)

```sh
# 1. Compile the kernel to an AIR object.
xcrun -sdk macosx metal -c kernel.metal \
    -I ../../dist \
    -o kernel.air

# 2. Link to a metallib.
xcrun -sdk macosx metallib kernel.air -o kernel.metallib

# 3. Compile and link the host program.
clang++ -std=c++17 -ObjC++ \
    -framework Metal -framework Foundation \
    main.mm -o standalone-msl

# 4. Run.
./standalone-msl
```

The host loads `kernel.metallib`, fills two `uint64_t` input buffers
with f64 bit patterns, dispatches `compute_fadd`, reads back the
results, and verifies them against host-side `double` arithmetic.

This example is shipped as documentation; it is not built by `cargo
test` (it has no Rust component). Its purpose is to prove out the
"drop-in MSL header" claim from the dist README.
