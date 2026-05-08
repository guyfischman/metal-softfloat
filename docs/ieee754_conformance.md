# IEEE-754 binary64 conformance

This document tracks the conformance status of `metal-softfloat`
against IEEE-754-2019 binary64 for arithmetic operations, the fused
multiply-add, and conversions. Tests live in
[`src/softfloat_ref.rs`](../src/softfloat_ref.rs) and
[`src/gpu.rs`](../src/gpu.rs); this doc summarises what's covered and
what's a known limitation.

## Operations covered

| Op | CPU (`cpu`) | softfloat_ref | MSL (GPU) |
|---|---|---|---|
| fadd / fsub / fmul | ✅ full domain | ✅ full domain | ✅ full domain |
| fdiv | ✅ full domain | ✅ full domain | ✅ full domain |
| fsqrt | ✅ full domain | ✅ full domain | ✅ full domain |
| fma | ✅ full domain | ✅ full domain | ✅ full domain |
| cvt i64/u64 ↔ f64 | ✅ | ✅ | ✅ |
| cvt f32 ↔ f64 | ✅ | ✅ | ✅ |
| comparisons (feq/lt/le/gt/ge) | ✅ | ✅ | ✅ |

The ✅ entries cross-check against native `f64` (or `f64::mul_add`) on
random inputs spanning all u64 bit patterns × all four rounding modes.

## §6 special operands

| Rule | CPU + softfloat_ref + MSL |
|---|---|
| §6.1 NaN propagation: `NaN op anything → NaN` | ✅ Canonical qNaN `0x7FF8_0000_0000_0000` returned for every NaN result |
| §6.2 ∞ + ∞ same sign → ∞; ∞ − ∞ → qNaN | ✅ |
| §6.2 ∞ × 0 → qNaN | ✅ |
| §6.2 ∞ / ∞ → qNaN; 0 / 0 → qNaN | ✅ |
| §6.2 finite / 0 → ±∞ (sign from XOR) | ✅ |
| §6.3 Sign of zero: `(+0) − (+0) = +0` (Nearest); `−0` under round-down | ✅ Tested explicitly in `fadd_special_cases` |
| §6.3 sqrt(±0) preserves sign; sqrt(−finite) → qNaN; sqrt(+∞) = +∞; sqrt(−∞) → qNaN | ✅ Tested in `fsqrt_special_cases` |

## §7 default exception handling

Exception flags are not exposed (no `fenv`-like inspection). Default
results are produced as IEEE-754 §7.4 specifies:

- §7.2 Invalid: qNaN result, no flag.
- §7.3 Divide-by-zero: ±∞ result, no flag.
- §7.4 Overflow: ±∞ in nearest mode; in directed modes, the away-from-target
  result is FLT_MAX (the largest representable finite) per the rounding rule.
- §7.5 Underflow: gradual underflow (subnormal output) is the default for
  every op (fadd / fsub / fmul / fdiv / fsqrt / fma) on both CPU and GPU.

## Rounding modes

All four of IEEE-754 §4 are implemented:

| Mode | `RoundingMode` | `mode: uint` (MSL) | ARM FPCR.RMode |
|---|---|---|---|
| Round-to-nearest-ties-to-even | `Nearest` | 0 | 00 |
| Round toward −∞ | `Down` | 1 | 10 |
| Round toward +∞ | `Up` | 2 | 01 |
| Round toward zero | `Zero` | 3 | 11 |

`cpu::*` uses ARM FPCR control via the `Guard` RAII type.
`softfloat_ref::*` and the MSL kernels take the mode as an explicit
argument and round in software.

## Signaling vs quiet NaN

Signaling-NaN distinction is **not preserved**. Every NaN result is
the canonical qNaN `0x7FF8_0000_0000_0000`. Most softfloat libraries
make this same simplification because:

- IEEE-754 §6.2 already permits payload propagation to be implementation-
  defined (every implementation must produce a qNaN; payload is optional).
- Distinguishing sNaN from qNaN at every entry point doubles the special-
  case dispatch cost without affecting any caller we know of.
- The Berkeley SoftFloat reference implementation also collapses sNaN
  payloads in some configurations (e.g. without `softfloat_propagateNaN_F64UI`).

If signaling NaN preservation is needed, file an issue.

## Limitations / future work

1. **Exception flags not exposed.** No `fenv`-style API for invalid /
   overflow / underflow / inexact / divide-by-zero flags. Most consumers
   typical consumers (ML frameworks, MD engines) don't use them; if needed, add a thread-local
   flag set to `cpu` and an output buffer for the GPU kernels.

## Verification commands

```bash
# All in-tree tests.
cargo test --lib

# Just the conformance fuzz (lib tests with full-domain inputs).
cargo test --lib full_domain

# GPU↔CPU cross-check.
cargo test --lib gpu::
```

## Berkeley TestFloat (opt-in)

The in-tree fuzz cross-checks against native `f64`, which is good for
catching gross errors but under-samples rounding-boundary, sticky-bit,
and subnormal-cliff cases. Berkeley TestFloat
(http://www.jhauser.us/arithmetic/TestFloat.html) generates targeted
vectors that exercise exactly those edges, and is the canonical
reference suite for softfloat implementations.

The `testfloat` Cargo feature wires `softfloat_ref` into a TestFloat
harness at [`tests/testfloat_conformance.rs`](../tests/testfloat_conformance.rs).
It's off by default because it depends on the external `testfloat_gen`
binary.

### Install TestFloat-3e

TestFloat-3e depends on Berkeley SoftFloat-3e — build SoftFloat first,
then TestFloat. The `Linux-x86_64-GCC` template builds cleanly on
macOS arm64 with the system clang (invoked as `gcc`):

```bash
# SoftFloat-3e
curl -O http://www.jhauser.us/arithmetic/SoftFloat-3e.zip
unzip SoftFloat-3e.zip
( cd SoftFloat-3e/build/Linux-x86_64-GCC && make )

# TestFloat-3e (must sit alongside SoftFloat-3e/)
curl -O http://www.jhauser.us/arithmetic/TestFloat-3e.zip
unzip TestFloat-3e.zip
( cd TestFloat-3e/build/Linux-x86_64-GCC && make )

# `testfloat_gen` is now at TestFloat-3e/build/Linux-x86_64-GCC/testfloat_gen
```

Put `testfloat_gen` on `PATH`, or pass its absolute path via the
`TESTFLOAT_GEN` env var.

### Run the conformance suite

```bash
# Fast (level 1) — ~tens of thousands of vectors per op × mode.
cargo test --release --features testfloat \
    --test testfloat_conformance

# Thorough (level 2) — millions of vectors per op × mode.
TESTFLOAT_LEVEL=2 cargo test --release \
    --features testfloat --test testfloat_conformance
```

The harness has 24 leaf `#[test]`s — one per (op, rounding mode) —
covering `f64_add`, `f64_sub`, `f64_mul`, `f64_div`, `f64_sqrt`, and
`f64_mulAdd`. Cargo runs `#[test]` functions in parallel by default
(`--test-threads = num_cpus`), so on a multi-core machine the slow
op (fma) ends up running its four rounding modes concurrently.

NaN payload differences are tolerated (every NaN result canonicalizes
to qNaN, see "Signaling vs quiet NaN" above). Status flags are not
checked since this crate doesn't expose them.

Last verified at level 1 against TestFloat-3e: ~25M vectors total
across all six ops × four rounding modes, zero mismatches.

The MSL kernels in `softfloat64.metal` are bit-identical translations
of `softfloat_ref` — passing TestFloat on the Rust side implies the
same conformance on the GPU side, modulo any future divergence. For a
*direct* GPU-side measurement see the next section.

The `ftz` and `testfloat` features are mutually exclusive:
TestFloat's vectors assume IEEE-754 §7.4 gradual underflow, which FTZ
violates by construction.

## Berkeley TestFloat against the MSL kernels (opt-in, macOS)

The `testfloat-gpu` feature drives the same Berkeley TestFloat vectors
**directly through the Metal kernels** — not transitively via
`softfloat_ref`. It vendors SoftFloat-3e + TestFloat-3e case generators
under [`vendor/`](../vendor) and links them via [`build.rs`](../build.rs):
inputs come from TestFloat's case generator, expected outputs from
SoftFloat-3e (the canonical reference), and both are batched into the
existing `gpu::*_batch` MSL dispatchers. No `testfloat_gen` subprocess,
no pipe — vectors flow at memory bandwidth and the GPU is the only
non-trivial cost.

Run it:

```bash
# Level 1 — ~25M vectors, runs in seconds.
cargo test --release --features testfloat-gpu \
    --test testfloat_gpu_conformance

# Level 2 — ~670M vectors, runs in seconds with nextest, ~tens of
# minutes with plain `cargo test` (single-process serialization).
TESTFLOAT_LEVEL=2 cargo nextest run --release \
    --features testfloat-gpu --test testfloat_gpu_conformance
```

The harness has 24 leaf tests — one per (op × rounding mode). Within a
single process they share C globals (TestFloat case-gen state +
SoftFloat's `softfloat_roundingMode`) and serialize on a mutex. With
[`cargo-nextest`](https://nexte.st) each `#[test]` runs in its own
process, so the mutex is a no-op and we get real CPU parallelism on
case generation and SoftFloat reference computation; the GPU command
queue still serializes at the OS level, which is the actual ceiling.

Throughput on M-series Apple Silicon (14 cores, 14 GPU cores):

| Level | Vectors | `cargo test` | `cargo nextest run` |
|---|---:|---:|---:|
| 1 | ~25M  | 3.0 s | 1.3 s |
| 2 | ~670M | (impractical¹) | 6.7 s |

¹ Level 2 fma is `2 × f64NumQInP2³` = ~181 billion vectors per rounding
mode (×4 modes ≈ 723 B). Even bare case-generator iteration would take
hours, before any expected-output computation. The harness caps fma at
TestFloat level 1 (6.13 M / mode) by default; override via
`TESTFLOAT_FMA_LEVEL=2` if you have time. Level 2 for the 2-arg ops
(`f64_add`, `f64_sub`, `f64_mul`, `f64_div`) and `f64_sqrt` runs at
the requested level unchanged.

Last verified: all 24 tests pass at level 2 (~670M vectors total)
directly through the MSL kernels, zero mismatches.

The `ftz` and `testfloat-gpu` features are mutually exclusive for the
same reason: TestFloat assumes IEEE-754 §7.4 gradual underflow.
