//! Soft-float f64 for Metal.
//!
//! Apple GPUs are f32-only: Metal has no `double` type, no f64 math
//! instructions, no hardware f64 registers. Any computation that
//! requires IEEE-754 double precision on GPU needs software emulation.
//!
//! This crate provides two matching APIs:
//!
//! - [`softfloat_ref`]: pure-integer Rust f64 emulation. Bit-exact
//!   with the MSL kernels and portable across architectures, so it
//!   doubles as a deterministic-f64 reference for non-Apple hosts.
//! - [`gpu`]: Metal-backed implementations. Each op round-trips
//!   through a Metal kernel that computes the result using integer
//!   arithmetic on the f64 bit pattern.
//!
//! ## Production use: include the MSL header, don't call the `gpu` module
//!
//! The `gpu` module is intended primarily for **testing, fuzzing, and
//! benchmarking** the MSL kernels from Rust — that's why it lives behind
//! a non-default feature flag. Each call:
//!
//! - allocates fresh Metal buffers and a one-shot command encoder,
//! - dispatches a single kernel,
//! - **blocks the host** on `wait_until_completed` before returning, and
//! - on the very first call, synchronously compiles ~95 KB of MSL source
//!   from a string (the Apple driver does this — expect a multi-100ms
//!   stall before your first `*_batch` returns).
//!
//! For real GPU code, prefer the redistributable header at
//! `dist/softfloat64.metal` (also exposed as the [`METAL_SOURCE`]
//! constant): `#include` it from your own MSL kernel and call
//! `__softfloat64_*` directly inline, alongside whatever else the kernel
//! is already doing. That keeps the softfloat work on-device, hidden
//! behind your own existing Metal pipeline + command-buffer setup, and
//! avoids a per-batch host↔device round-trip.
//!
//! There's also an internal `cpu` module (test-only) that wraps native
//! `f64` operations with FPCR rounding-mode control on aarch64 — used
//! as the directed-rounding ground truth for the in-tree fuzz tests
//! against `softfloat_ref`. It's not part of the public API and is
//! `#[cfg(test)]` so it doesn't ship in the compiled crate.
//!
//! All inputs and outputs are `u64` — the IEEE-754 bit pattern of an
//! f64. Consumers that already hold `f64` values can convert with
//! `f64::to_bits` / `f64::from_bits`.
//!
//! ## Rounding modes
//!
//! Each op takes an explicit [`RoundingMode`]:
//!
//! - [`RoundingMode::Nearest`] (ties to even) — IEEE default
//! - [`RoundingMode::Down`] (toward −∞)
//! - [`RoundingMode::Up`] (toward +∞)
//! - [`RoundingMode::Zero`] (truncate)
//!
//! ## `no_std`
//!
//! `softfloat_ref` and the `RoundingMode` enum are pure `core` — no
//! heap, no allocator, no `std`. The crate sets `#![no_std]` for non-
//! test, non-`gpu` builds, so embedded targets, kernels, and other
//! `no_std` consumers (lockstep simulators, deterministic execution
//! environments, on-chain WASM runtimes) get a deterministic-f64
//! reference implementation by default. The `gpu` feature opts back
//! into `std` because the Metal dispatchers need `HashMap` / `RwLock`.

#![cfg_attr(not(test), no_std)]
#![allow(clippy::missing_panics_doc)]

// `gpu` needs std (HashMap, RwLock, OnceLock, Vec, Instant). Bring it
// in when the feature is active and we're on macOS. Other build
// configurations stay no_std.
#[cfg(all(feature = "gpu", target_os = "macos"))]
extern crate std;

/// IEEE-754 f64 rounding mode. The numeric values are stable across
/// the Rust API and the MSL kernels (the kernel takes `mode` as `uint`),
/// so a host-side `RoundingMode as u32` round-trips cleanly through a
/// shader argument. The encoding is also compatible with RandomX's
/// `CFROUND` instruction for callers who care.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundingMode {
    Nearest = 0,
    Down = 1,
    Up = 2,
    Zero = 3,
}

// `cpu` is test-only: it pokes the FP control register on aarch64 to
// validate directed-rounding ops against `softfloat_ref`. Mutating
// global FPCR state from a library is unsafe in general (LLVM doesn't
// model the rounding mode in its IR, so it can hoist FP ops across the
// guard), so this stays gated behind `#[cfg(test)]` and never ships in
// the compiled crate. Anyone wanting cross-platform deterministic
// directed rounding should use `softfloat_ref::*` instead — it does
// the rounding in software and is bit-identical on every target.
#[cfg(test)]
mod cpu;
// The `gpu` module wraps Apple Metal, which only exists on macOS. We
// therefore gate the module on (feature, OS): a Linux user enabling
// `--features gpu` still gets a successful build with just the
// `softfloat_ref` reference implementation.
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub mod gpu;
pub mod softfloat_ref;

/// The redistributable MSL source for `softfloat64.metal`, embedded as
/// a string at compile time.
///
/// Use this when integrating with `metal-rs` (or any other Rust Metal
/// binding) to avoid reading the file from disk:
///
/// ```ignore
/// use metal::{CompileOptions, Device};
/// use metal_softfloat::METAL_SOURCE;
///
/// let device = Device::system_default().unwrap();
/// let library = device
///     .new_library_with_source(METAL_SOURCE, &CompileOptions::new())
///     .unwrap();
/// ```
///
/// The source exposes the public `__softfloat64_*` API
/// (`__softfloat64_fadd`, `_fsub`, `_fmul`, `_fdiv`, `_fsqrt`, `_fma`).
/// Define `SOFTFLOAT_FTZ` at compile time (`-DSOFTFLOAT_FTZ`) to flush
/// subnormal inputs and outputs to zero.
pub const METAL_SOURCE: &str = include_str!("../dist/softfloat64.metal");
