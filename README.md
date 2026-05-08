# `metal-softfloat-core` — IEEE-754 binary64 for Metal (and Rust)

Two parallel implementations of f64 arithmetic, both built from
integer-only operations on the IEEE-754 bit pattern:

- **`dist/softfloat64.metal`** — a self-contained Metal Shading Language
  header. Drop into any Apple Metal compute kernel; `#include` it and
  call the `__softfloat64_*` functions on `ulong` bit patterns. No host
  dependency.
- **`metal_softfloat_core::softfloat_ref`** — pure-Rust f64 emulation,
  bit-exact with the MSL implementation. Useful when you need the same
  deterministic f64 result on a non-Apple host (lockstep simulators,
  consensus-critical math, GPU-vs-CPU validation).

## What's inside

The standard IEEE-754 path:

```
ulong __softfloat64_fadd(ulong a, ulong b, uint mode);
ulong __softfloat64_fsub(ulong a, ulong b, uint mode);
ulong __softfloat64_fmul(ulong a, ulong b, uint mode);
ulong __softfloat64_fdiv(ulong a, ulong b, uint mode);
ulong __softfloat64_fsqrt(ulong a, uint mode);
ulong __softfloat64_fma(ulong a, ulong b, ulong c, uint mode);
```

`a`, `b`, `c` are IEEE-754 binary64 bit patterns (cast from `double`
on the host with `*(uint64_t*)&x`, or `as_type<ulong>` on a hypothetical
f64 inside MSL — Apple's MSL has no native `double`, which is why this
file exists).

Conversions between f64 and integer / f32 types:

```
ulong __softfloat64_cvt_i64_to_f64(long  x, uint mode);
ulong __softfloat64_cvt_u64_to_f64(ulong x, uint mode);
long  __softfloat64_cvt_f64_to_i64(ulong a, uint mode);  // saturates; NaN→0
ulong __softfloat64_cvt_f32_to_f64(uint  a);             // exact, no rmode
uint  __softfloat64_cvt_f64_to_f32(ulong a, uint mode);
```

IEEE-754 §5.11 quiet comparisons (any-NaN → false):

```
bool __softfloat64_feq(ulong a, ulong b);
bool __softfloat64_flt(ulong a, ulong b);
bool __softfloat64_fle(ulong a, ulong b);
bool __softfloat64_fgt(ulong a, ulong b);
bool __softfloat64_fge(ulong a, ulong b);
```

Plus an "unpacked" path for tight inner loops:

```
struct __softfloat64_unp { ulong sign; int exp; ulong mantissa; };

__softfloat64_unp __softfloat64_unpack(ulong bits);
ulong              __softfloat64_pack  (__softfloat64_unp u, uint mode);

__softfloat64_unp __softfloat64_unp_fadd (__softfloat64_unp a, __softfloat64_unp b, uint mode);
__softfloat64_unp __softfloat64_unp_fsub (__softfloat64_unp a, __softfloat64_unp b, uint mode);
__softfloat64_unp __softfloat64_unp_fmul (__softfloat64_unp a, __softfloat64_unp b, uint mode);
__softfloat64_unp __softfloat64_unp_fdiv (__softfloat64_unp a, __softfloat64_unp b, uint mode);
__softfloat64_unp __softfloat64_unp_fsqrt(__softfloat64_unp a, uint mode);
__softfloat64_unp __softfloat64_unp_fma  (__softfloat64_unp a, __softfloat64_unp b,
                                          __softfloat64_unp c, uint mode);
```

The `_unp_*` family skips the per-call special-case dispatch (NaN /
±Inf / ±0 / subnormal handling) and the per-call pack/unpack churn.
Pre-unpack once with `__softfloat64_unpack`, run many ops on the
unpacked state, repack once with `__softfloat64_pack` at the end.

**Caller must guarantee normal inputs throughout the loop body.**
Non-normal operands silently produce wrong answers — this path has
no special-case dispatch. See the reduction sketch under "Usage" below.

Plus throughput kernels for benchmarking your hardware:

```
kernel void __softfloat64_fadd_chain (constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_fsub_chain (constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_fmul_chain (constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_fdiv_chain (constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_fsqrt_chain(constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_fma_chain  (constant ulong2& seed, device ulong* out, ...);

kernel void __softfloat64_cvt_i64_to_f64_chain(constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_cvt_u64_to_f64_chain(constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_cvt_f64_to_i64_chain(constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_cvt_f32_to_f64_chain(constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_cvt_f64_to_f32_chain(constant ulong2& seed, device ulong* out, ...);

kernel void __softfloat64_feq_chain(constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_flt_chain(constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_fle_chain(constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_fgt_chain(constant ulong2& seed, device ulong* out, ...);
kernel void __softfloat64_fge_chain(constant ulong2& seed, device ulong* out, ...);
```

Each kernel runs `__SOFTFLOAT64_CHAIN_OPS` (1024) chained ops per
thread with a cheap mantissa-twiddle chain-breaker, kept in the normal
fast path so you measure the FPU instead of NaN/Inf branch costs.
Total ops dispatched = `threads × 1024`. Measured Apple Silicon (M4
Pro) results vs a 14-thread CPU hardware-f64 baseline (see
`examples/throughput_demo.rs`):

| op             | CPU 14T (hw f64) | GPU softfloat | speedup |
|----------------|------------------|---------------|---------|
| fadd           |  3.07 G/s        | 18.55 G/s     |  6.0×   |
| fsub           |  2.90 G/s        | 18.14 G/s     |  6.3×   |
| fmul           |  2.72 G/s        | 20.90 G/s     |  7.7×   |
| fdiv           |  1.96 G/s        | 15.71 G/s     |  8.0×   |
| fsqrt          |  1.80 G/s        | 19.88 G/s     | 11.0×   |
| fma            |  2.86 G/s        | 12.69 G/s     |  4.4×   |
| cvt_i64_to_f64 |  2.93 G/s        | 42.36 G/s     | 14.5×   |
| cvt_u64_to_f64 |  3.03 G/s        | 48.46 G/s     | 16.0×   |
| cvt_f64_to_i64 |  2.90 G/s        | 29.64 G/s     | 10.2×   |
| cvt_f32_to_f64 |  3.03 G/s        | 78.20 G/s     | 25.8×   |
| cvt_f64_to_f32 |  3.03 G/s        | 31.73 G/s     | 10.5×   |
| feq            |  2.75 G/s        | 98.68 G/s     | 35.9×   |
| flt            |  2.68 G/s        | 63.84 G/s     | 23.8×   |
| fle            |  2.92 G/s        | 71.95 G/s     | 24.6×   |
| fgt            |  2.86 G/s        | 85.07 G/s     | 29.7×   |
| fge            |  2.69 G/s        | 81.67 G/s     | 30.4×   |

GPU softfloat outputs are bit-equal to the corresponding CPU hardware
f64 results under round-to-nearest, so this is a faster path that's
also protocol-equivalent — same 64 bits, more throughput.

`mode` is the rounding mode:

| `mode` | meaning |
|---|---|
| `0` | round-to-nearest-ties-to-even (IEEE default) |
| `1` | round toward −∞ |
| `2` | round toward +∞ |
| `3` | round toward zero |

If you don't care about non-default rounding, pass `0`.

## IEEE-754 conformance

- All four rounding modes
- NaN / ±Inf / ±0 propagation per §6
- Subnormal inputs and outputs (gradual underflow) for every op:
  fadd / fsub / fmul / fdiv / fsqrt / fma
- `__softfloat64_fma` is single-rounding (no intermediate rounding of `a × b`)
- Conversions and comparisons cross-checked GPU↔softfloat_ref bit-for-bit
  on full-domain inputs (i64::MIN, mantissa cliffs, ±0, ±Inf, NaN, f32
  subnormal/overflow boundaries × all four rounding modes)
- Canonical qNaN `0x7FF8_0000_0000_0000` for every NaN result; signaling-NaN
  payloads are not preserved

See [`docs/ieee754_conformance.md`](docs/ieee754_conformance.md) for the
operation-by-operation conformance matrix.

## Usage

### From an MSL kernel

```metal
#include <metal_stdlib>
using namespace metal;

#include "softfloat64.metal"

kernel void energy_step(
    device const ulong* a       [[buffer(0)]],
    device const ulong* b       [[buffer(1)]],
    device ulong*       energy  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    // (a × b) + energy, with one rounding (Kahan-like accumulation).
    energy[gid] = __softfloat64_fma(a[gid], b[gid], energy[gid], 0u);
}
```

### Tight reductions (the unpacked path)

For per-thread reductions (sum, mean, dot product, layer-norm /
softmax denominator, attention scores) the dominant cost is *not* the
arithmetic — it's the per-call special-case dispatch and pack/unpack
that surrounds it. The `_unp_*` family lets you pay that once at the
loop head and once at the loop tail, then keep the working state in
unpacked form across every iteration:

```metal
#include "softfloat64.metal"

kernel void kahan_sum(
    device const ulong* contribs     [[buffer(0)]],   // f64 bit patterns
    device ulong*       partials     [[buffer(1)]],
    constant uint&      n_per_thread [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint base = gid * n_per_thread;
    __softfloat64_unp acc = __softfloat64_unpack(0u);
    __softfloat64_unp c   = __softfloat64_unpack(0u);

    for (uint i = 0; i < n_per_thread; ++i) {
        __softfloat64_unp x = __softfloat64_unpack(contribs[base + i]);
        __softfloat64_unp y = __softfloat64_unp_fsub(x, c, 0u);
        __softfloat64_unp t = __softfloat64_unp_fadd(acc, y, 0u);
        // Kahan compensation: c = (t - acc) - y
        __softfloat64_unp d = __softfloat64_unp_fsub(t, acc, 0u);
        c   = __softfloat64_unp_fsub(d, y, 0u);
        acc = t;
    }
    partials[gid] = __softfloat64_pack(acc, 0u);
}
```

The trade-off is correctness: the `_unp_*` family assumes every
operand stays a normal f64 throughout the loop body. If your input
data can contain NaN / ±Inf / ±0 / subnormals you have to either
filter upstream or stick with the regular `__softfloat64_*` family,
which costs ~5 extra branches per call.

### Compile

```sh
xcrun -sdk macosx metal -c your_kernel.metal -I path/to/dist/ -o your_kernel.air
xcrun -sdk macosx metallib your_kernel.air -o your_kernel.metallib
```

Add `-DSOFTFLOAT_FTZ` to the `metal` invocation to flush subnormal
inputs and outputs to zero (matches the `ftz` Cargo feature on the
Rust side).

### Host-side (Objective-C / Swift / Rust + `metal-rs`)

Treat `ulong` buffers as `uint64_t` arrays of f64 bit patterns. Convert
to/from `double` with `__bit_cast<uint64_t>(x)` (C++) or
`x.to_bits()` / `f64::from_bits(b)` (Rust).

### From Rust + `metal-rs`

The MSL source is also exposed as a Rust constant, so consumers don't
need to read the file off disk:

```rust
use metal::{CompileOptions, Device};
use metal_softfloat_core::METAL_SOURCE;

let device = Device::system_default().unwrap();
let library = device
    .new_library_with_source(METAL_SOURCE, &CompileOptions::new())
    .unwrap();
```

#### A note on the `gpu` Cargo feature

The `gpu` feature gates a `metal_softfloat_core::gpu` module that wraps
each `__softfloat64_*` kernel in a Rust `*_batch` / `*_chain` helper.
**Those helpers exist for this crate's tests, fuzzers, and benchmarks
— not as a general-purpose GPU API.** Each call:

- allocates fresh Metal buffers and a one-shot command encoder,
- dispatches one kernel,
- **blocks the calling thread** on `cmd.wait_until_completed()`, and
- pays a multi-100 ms first-call MSL compile (the driver caches the
  result for the rest of the process).

For production code, `#include "softfloat64.metal"` (or the
`METAL_SOURCE` constant) into your own MSL kernels and call
`__softfloat64_*` inline alongside the work you're already doing on the
GPU. Keep the dispatch loop, command-buffer reuse, and synchronization
strategy on your side — the softfloat header makes no assumptions about
any of those.

## Pure-Rust API

The `softfloat_ref` module re-exposes the same algorithms as ordinary
Rust functions taking `u64` bit patterns and a `RoundingMode`. Because
the implementation is integer-only, the result is identical on every
platform — useful for lockstep / consensus-critical math:

```rust
use metal_softfloat_core::{softfloat_ref, RoundingMode};

let x = 1.0_f64.to_bits();
let y = 2.0_f64.to_bits();
let sum = softfloat_ref::fadd(x, y, RoundingMode::Nearest);
assert_eq!(f64::from_bits(sum), 3.0);
```

`softfloat_ref` covers fadd / fsub / fmul / fdiv / fsqrt / fma plus
i64↔f64, u64↔f64, f32↔f64 conversions and the IEEE comparisons (feq /
flt / fle / fgt / fge). The MSL kernels expose the same surface — the
`__softfloat64_cvt_*` and `__softfloat64_f{eq,lt,le,gt,ge}` functions
are bit-exact with `softfloat_ref` on every input.

## Audience

`softfloat64.metal` gives you **bit-exact IEEE-754 binary64** on Apple
GPU — same operation sequence, same rounding decisions, *same 64
output bits* as a CPU `double` reference. That property matters when:

- **Consensus / on-chain f64.** EVM precompiles
  that quote IEEE-754, deterministic smart contracts. The protocol
  fork-rule says "f64 results match the reference bit-for-bit"; nodes
  on a mix of CPU and Apple GPU all need to agree to the last ulp.
- **Lockstep simulators / multiplayer games** where state must evolve
  identically across CPU and GPU clients.
- **GPU-vs-CPU validation harnesses** for porting an f64 codebase —
  confirming the GPU port produces exactly what the CPU reference
  produced before swapping it in.
- **f64-on-Apple-GPU at higher throughput than CPU multi-core** (see
  the table above; 5-12× win on most ops). Apple GPUs have no f64
  hardware, so the obvious assumption is "Apple GPU can't do f64
  fast." That's true for any *single* op, but aggregate softfloat
  throughput across the GPU's many lanes beats the CPU's combined
  hardware-FPU throughput once your kernel keeps enough state on the
  GPU per dispatch. The `*_chain` kernels measure this directly.

If you need "more precision than f32, as fast as possible" rather than
specifically "bit-equal to f64", look at
[metal-softfloat-pytorch](https://github.com/guyfischman/metal-softfloat-pytorch)
instead — it uses triple-float (~72 effective mantissa bits, ~10
native f32 ops per add) which is *faster than f64 softfloat* and *more
accurate than f64*, but it's not bit-equal to a CPU `double` reference.

## Example programs

- [`examples/standalone-msl/`](examples/standalone-msl/) — a small
  Metal C++ command-line tool that loads `softfloat64.metal`, runs a
  batch of `__softfloat64_fadd` calls on a buffer of bit patterns, and
  prints the bit-exact results. No Rust dependency.
- [`examples/throughput_demo.rs`](examples/throughput_demo.rs) — runs
  the public `gpu::*_chain` API for each op, dispatching ~200M
  softfloat ops per measurement; compares against a 14-thread
  hardware-f64 CPU baseline doing the same op count. Covers all 16
  ops (arithmetic, conversions, comparisons). Also runs a 4096-pair
  `gpu::fadd_batch` correctness check against `softfloat_ref` to
  confirm the GPU kernels are bit-equal to native CPU f64 in nearest
  mode. Source for the throughput table above.

## Versioning

`dist/softfloat64.metal` is generated from `shaders/softfloat.metal` by
`cargo run --bin gen-msl-header`. CI verifies
the generated content matches the checked-in copy
(`cargo run --bin gen-msl-header -- --check`).

## License

Dual-licensed:

- The Berkeley SoftFloat reciprocal / reciprocal-sqrt approximation
  tables and the inline functions derived from them
  (`approx_recip32_1`, `approx_recip_sqrt32_1`, `softfloat_div`,
  `softfloat_sqrt`) are BSD-3-Clause from Berkeley SoftFloat-3e
  (Copyright 2011–2017 The Regents of the University of California).
- All other contributions are MIT.

See the full text in `softfloat64.metal`'s header.
