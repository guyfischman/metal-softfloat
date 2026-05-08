//! Metal-backed implementations of the f64 ops.
//!
//! **Intended audience: this crate's own tests, fuzzers, and benchmarks.**
//! For production GPU code, prefer `#include "softfloat64.metal"`
//! (`crate::METAL_SOURCE`) directly from your MSL kernels and call
//! `__softfloat64_*` inline — that keeps the softfloat work on-device
//! behind your existing Metal pipeline.
//!
//! What you get if you call this module from production code:
//!
//! - **Blocking dispatch.** Every `*_batch` / `*_chain` waits on
//!   `cmd.wait_until_completed()` before returning. The CPU thread is
//!   parked for the duration of the GPU launch.
//! - **Per-call buffer churn.** Each dispatch allocates fresh input /
//!   output buffers and a one-shot command encoder; nothing is reused
//!   across calls. Fine for batches in the 10⁵–10⁷ range, terrible at
//!   single-op latency.
//! - **First-call MSL compilation.** The shader source is ~95 KB
//!   compiled at runtime from a string the first time `ctx()` is hit.
//!   Apple's driver caches the result, but expect the first GPU call in
//!   a process to take noticeably longer than subsequent ones.
//!
//! Each op launches a compute kernel that runs one thread per input,
//! applying the softfloat algorithm in pure integer arithmetic.

// Crate-level `#![no_std]` means `std`'s prelude isn't implicit here;
// pull in the bits this module needs.
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::vec;
use std::vec::Vec;

use metal::{
    CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};

use super::RoundingMode;

// Compile our own GPU dispatchers against the test-kernel-enabled
// variant of the shader. The `#define` activates the
// `#ifdef METAL_SOFTFLOAT_TESTS` block in `softfloat.metal`. Public
// consumers `#include`-ing the same file (or `dist/softfloat64.metal`)
// without that define get just the public `__softfloat64_*` API and
// pay no compile cost for the test kernels.
//
// The "proper" Metal way to do this is `MTLCompileOptions
// preprocessorMacros = @{ @"METAL_SOFTFLOAT_TESTS": @1 }`, exposed by
// metal-rs as `CompileOptionsRef::set_preprocessor_macros(*mut Object)`.
// That takes a hand-rolled `NSDictionary<NSString, NSNumber>` we'd
// have to construct via raw `objc::msg_send!` calls — metal-rs
// deliberately leaves NSDictionary marshaling as an exercise (the
// crate's own source even has a `// TODO: figure out NSDictionary
// wrapper` next to a similar API). Prepending a `#define` to the
// source is a perfectly valid C++/MSL pattern and saves us all that.
const SHADER_SRC: &str = concat!(
    "#define METAL_SOFTFLOAT_TESTS 1\n",
    include_str!("../shaders/softfloat.metal"),
);

// SAFETY: We hold these Metal objects behind a global OnceLock; Apple's
// Metal types are thread-safe in practice. The pipeline cache uses an
// RwLock so concurrent dispatches share a read lock on the hot (cached)
// path; only first-time pipeline creation takes the write lock.
struct GpuCtx {
    device: Device,
    library: Library,
    queue: CommandQueue,
    pipelines: RwLock<HashMap<&'static str, ComputePipelineState>>,
}

unsafe impl Send for GpuCtx {}
unsafe impl Sync for GpuCtx {}

impl GpuCtx {
    fn pipeline(&self, kernel_name: &'static str) -> ComputePipelineState {
        if let Some(p) = self.pipelines.read().expect("pipeline rwlock").get(kernel_name) {
            return p.clone();
        }
        let function = self
            .library
            .get_function(kernel_name, None)
            .expect("get fn");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");
        self.pipelines
            .write()
            .expect("pipeline rwlock")
            .insert(kernel_name, pipeline.clone());
        pipeline
    }

}

fn ctx() -> &'static GpuCtx {
    static CTX: OnceLock<GpuCtx> = OnceLock::new();
    CTX.get_or_init(|| {
        let device = Device::system_default().expect("no Metal device");
        let library = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile softfloat.metal");
        let queue = device.new_command_queue();
        GpuCtx {
            device,
            library,
            queue,
            pipelines: RwLock::new(HashMap::new()),
        }
    })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Op2Input {
    a: u64,
    b: u64,
    mode: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Op1Input {
    a: u64,
    mode: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Op3Input {
    a: u64,
    b: u64,
    c: u64,
    mode: u32,
    _pad: u32,
}

// Common launch shape shared by every kernel in this module: one thread
// per output, output buffer at index 1, readback into a `Vec<u64>`. The
// only thing that varies between kernels is what gets bound at index 0
// (a per-thread input buffer for batch ops, a uniform seed for chain
// kernels). `bind_input` is a one-shot closure so each call site can
// bind whatever it needs without a second helper.
fn launch<F>(kernel_name: &'static str, threads: usize, bind_input: F) -> Vec<u64>
where
    F: FnOnce(&ComputeCommandEncoderRef),
{
    if threads == 0 {
        return Vec::new();
    }
    let c = ctx();
    let pipeline = c.pipeline(kernel_name);
    let out_buf = c.device.new_buffer(
        (threads * core::mem::size_of::<u64>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = c.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    bind_input(encoder);
    encoder.set_buffer(1, Some(&out_buf), 0);

    let tg = pipeline.thread_execution_width();
    let grid = MTLSize::new(threads as u64, 1, 1);
    let threadgroup = MTLSize::new(tg.min(threads as u64), 1, 1);
    encoder.dispatch_threads(grid, threadgroup);
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let mut out = vec![0u64; threads];
    unsafe {
        core::ptr::copy_nonoverlapping(
            out_buf.contents().cast::<u64>(),
            out.as_mut_ptr(),
            threads,
        );
    }
    out
}

// Per-thread-input dispatch. `T` is the input record laid out as a `repr(C)`
// struct that matches the kernel's input slot; we just upload the slice
// verbatim and let the kernel index into it by `gid`.
fn dispatch_kernel<T: Copy>(kernel_name: &'static str, inputs: &[T]) -> Vec<u64> {
    if inputs.is_empty() {
        return Vec::new();
    }
    let c = ctx();
    let in_buf = c.device.new_buffer_with_data(
        inputs.as_ptr().cast(),
        core::mem::size_of_val(inputs) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    launch(kernel_name, inputs.len(), |encoder| {
        encoder.set_buffer(0, Some(&in_buf), 0);
    })
}

fn op2_inputs(pairs: &[(u64, u64, RoundingMode)]) -> Vec<Op2Input> {
    pairs
        .iter()
        .map(|&(a, b, m)| Op2Input {
            a,
            b,
            mode: m as u32,
            _pad: 0,
        })
        .collect()
}

fn op1_inputs(pairs: &[(u64, RoundingMode)]) -> Vec<Op1Input> {
    pairs
        .iter()
        .map(|&(a, m)| Op1Input {
            a,
            mode: m as u32,
            _pad: 0,
        })
        .collect()
}

macro_rules! op2_batch {
    ($name:ident, $kernel:expr) => {
        #[must_use]
        pub fn $name(pairs: &[(u64, u64, RoundingMode)]) -> Vec<u64> {
            dispatch_kernel($kernel, &op2_inputs(pairs))
        }
    };
}

op2_batch!(fadd_batch, "test_fadd");
op2_batch!(fsub_batch, "test_fsub");
op2_batch!(fmul_batch, "test_fmul");
op2_batch!(fdiv_batch, "test_fdiv");

#[must_use]
pub fn fsqrt_batch(pairs: &[(u64, RoundingMode)]) -> Vec<u64> {
    dispatch_kernel("test_fsqrt", &op1_inputs(pairs))
}

#[must_use]
pub fn fma_batch(triples: &[(u64, u64, u64, RoundingMode)]) -> Vec<u64> {
    let inputs: Vec<Op3Input> = triples
        .iter()
        .map(|&(a, b, c, m)| Op3Input { a, b, c, mode: m as u32, _pad: 0 })
        .collect();
    dispatch_kernel("test_fma", &inputs)
}

// =============================================================================
// Conversions and comparisons.
//
// All inputs / outputs travel as integer bit patterns: f64 as u64, f32 as
// u32 (zero-extended into u64 in the buffer), i64 as u64 via two's-
// complement reinterpret, comparison results as u64 (0 or 1). The MSL test
// kernels use the same Op1Input / Op2Input layouts as the arithmetic ops,
// so dispatch reuses dispatch_op1 / dispatch_op2.
// =============================================================================

/// i64 → f64 with mode-controlled rounding.
#[must_use]
pub fn cvt_i64_to_f64_batch(pairs: &[(i64, RoundingMode)]) -> Vec<u64> {
    // Reinterpret each i64 as u64 (two's-complement bit pattern) so we can
    // reuse the Op1Input layout. The MSL kernel does the inverse cast.
    let inputs: Vec<Op1Input> = pairs
        .iter()
        .map(|&(x, m)| Op1Input { a: x as u64, mode: m as u32, _pad: 0 })
        .collect();
    dispatch_kernel("test_cvt_i64_to_f64", &inputs)
}

/// u64 → f64 with mode-controlled rounding.
#[must_use]
pub fn cvt_u64_to_f64_batch(pairs: &[(u64, RoundingMode)]) -> Vec<u64> {
    dispatch_kernel("test_cvt_u64_to_f64", &op1_inputs(pairs))
}

/// f64 → i64 with mode-controlled rounding. NaN → 0; out-of-range
/// saturates to `i64::MIN` / `i64::MAX`.
#[must_use]
pub fn cvt_f64_to_i64_batch(pairs: &[(u64, RoundingMode)]) -> Vec<i64> {
    let raw = dispatch_kernel("test_cvt_f64_to_i64", &op1_inputs(pairs));
    raw.into_iter().map(|x| x as i64).collect()
}

/// f32 → f64 (exact, no rounding mode). Pass each f32 as its `u32` bit
/// pattern (`f32::to_bits`).
#[must_use]
pub fn cvt_f32_to_f64_batch(inputs: &[u32]) -> Vec<u64> {
    // Mode is unused by the kernel; pass Nearest as a stable filler.
    let pairs: Vec<Op1Input> = inputs
        .iter()
        .map(|&x| Op1Input { a: u64::from(x), mode: 0, _pad: 0 })
        .collect();
    dispatch_kernel("test_cvt_f32_to_f64", &pairs)
}

/// f64 → f32 with mode-controlled rounding. Result is the f32 bit pattern.
#[must_use]
pub fn cvt_f64_to_f32_batch(pairs: &[(u64, RoundingMode)]) -> Vec<u32> {
    let raw = dispatch_kernel("test_cvt_f64_to_f32", &op1_inputs(pairs));
    // Each output is a u32 zero-extended into a u64; truncate back.
    raw.into_iter().map(|x| x as u32).collect()
}

macro_rules! cmp_batch {
    ($name:ident, $kernel:expr) => {
        #[doc = concat!("IEEE-754 ", stringify!($name), " batch. Returns one bool per input pair.")]
        #[must_use]
        pub fn $name(pairs: &[(u64, u64)]) -> Vec<bool> {
            // Comparison kernels ignore the mode field; pass Nearest as filler.
            let inputs: Vec<Op2Input> = pairs
                .iter()
                .map(|&(a, b)| Op2Input { a, b, mode: 0, _pad: 0 })
                .collect();
            dispatch_kernel($kernel, &inputs)
                .into_iter()
                .map(|x| x != 0)
                .collect()
        }
    };
}

cmp_batch!(feq_batch, "test_feq");
cmp_batch!(flt_batch, "test_flt");
cmp_batch!(fle_batch, "test_fle");
cmp_batch!(fgt_batch, "test_fgt");
cmp_batch!(fge_batch, "test_fge");

// =============================================================================
// Throughput kernels (`__softfloat64_*_chain` family in `softfloat64.metal`).
//
// Each thread runs `CHAIN_OPS_PER_THREAD` chained softfloat ops with a
// cheap mantissa-twiddle chain-breaker, so total ops dispatched per call =
// `threads × CHAIN_OPS_PER_THREAD`. The output `Vec<u64>` has length
// `threads`; `out[gid]` is the final accumulator value of thread `gid`.
//
// These exist because the host↔device buffer-transfer cost of a `*_batch`
// call hides the GPU's actual softfloat throughput. The chain kernels
// keep ~1k ops on the GPU per thread before any host I/O — that's where
// aggregate softfloat throughput beats CPU multi-core hardware f64 by
// 5-11× on Apple Silicon. See `examples/throughput_demo.rs`.
// =============================================================================

/// Chained ops per thread, baked into every `__softfloat64_*_chain`
/// kernel in `softfloat64.metal`. Total ops per call to a chain
/// dispatcher is `threads × CHAIN_OPS_PER_THREAD`.
pub const CHAIN_OPS_PER_THREAD: usize = 1024;

fn dispatch_chain(kernel_name: &'static str, threads: usize, seed: (u64, u64)) -> Vec<u64> {
    let seed_pair: [u64; 2] = [seed.0, seed.1];
    launch(kernel_name, threads, |encoder| {
        // Chain kernels take the seed as a uniform constant rather than a
        // per-thread input record, so it travels via `set_bytes` (small
        // inline payload) instead of a freshly allocated buffer.
        encoder.set_bytes(
            0,
            core::mem::size_of_val(&seed_pair) as u64,
            seed_pair.as_ptr().cast(),
        );
    })
}

macro_rules! chain_dispatcher {
    ($name:ident, $kernel:expr) => {
        #[must_use]
        pub fn $name(threads: usize, seed: (u64, u64)) -> Vec<u64> {
            dispatch_chain($kernel, threads, seed)
        }
    };
}

chain_dispatcher!(fadd_chain, "__softfloat64_fadd_chain");
chain_dispatcher!(fsub_chain, "__softfloat64_fsub_chain");
chain_dispatcher!(fmul_chain, "__softfloat64_fmul_chain");
chain_dispatcher!(fdiv_chain, "__softfloat64_fdiv_chain");
chain_dispatcher!(fsqrt_chain, "__softfloat64_fsqrt_chain");
chain_dispatcher!(fma_chain, "__softfloat64_fma_chain");

chain_dispatcher!(cvt_i64_to_f64_chain, "__softfloat64_cvt_i64_to_f64_chain");
chain_dispatcher!(cvt_u64_to_f64_chain, "__softfloat64_cvt_u64_to_f64_chain");
chain_dispatcher!(cvt_f64_to_i64_chain, "__softfloat64_cvt_f64_to_i64_chain");
chain_dispatcher!(cvt_f32_to_f64_chain, "__softfloat64_cvt_f32_to_f64_chain");
chain_dispatcher!(cvt_f64_to_f32_chain, "__softfloat64_cvt_f64_to_f32_chain");

chain_dispatcher!(feq_chain, "__softfloat64_feq_chain");
chain_dispatcher!(flt_chain, "__softfloat64_flt_chain");
chain_dispatcher!(fle_chain, "__softfloat64_fle_chain");
chain_dispatcher!(fgt_chain, "__softfloat64_fgt_chain");
chain_dispatcher!(fge_chain, "__softfloat64_fge_chain");


#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu;
    use rand::{Rng, SeedableRng};

    fn rand_normal(rng: &mut impl Rng) -> f64 {
        loop {
            let exp_shift: i32 = rng.gen_range(-30..30);
            let sign = if rng.gen::<bool>() { -1.0 } else { 1.0 };
            let x = sign * rng.gen::<f64>() * 2f64.powi(exp_shift);
            if x.is_normal() {
                return x;
            }
        }
    }

    fn gen_op2(seed: u64, n: usize) -> Vec<(u64, u64, RoundingMode)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let modes = [
            RoundingMode::Nearest,
            RoundingMode::Down,
            RoundingMode::Up,
            RoundingMode::Zero,
        ];
        (0..n)
            .map(|i| {
                let a = rand_normal(&mut rng);
                let b = rand_normal(&mut rng);
                (a.to_bits(), b.to_bits(), modes[i % 4])
            })
            .collect()
    }

    fn gen_op1(seed: u64, n: usize, positive: bool) -> Vec<(u64, RoundingMode)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let modes = [
            RoundingMode::Nearest,
            RoundingMode::Down,
            RoundingMode::Up,
            RoundingMode::Zero,
        ];
        (0..n)
            .map(|i| {
                let mut x = rand_normal(&mut rng);
                if positive {
                    x = x.abs();
                }
                (x.to_bits(), modes[i % 4])
            })
            .collect()
    }

    fn check_op2(name: &str, f_cpu: fn(u64, u64, RoundingMode) -> u64, f_gpu: fn(&[(u64, u64, RoundingMode)]) -> Vec<u64>, seed: u64) {
        let n: usize = std::env::var("METAL_SF_FUZZ_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let inputs = gen_op2(seed, n);
        let gpu_out = f_gpu(&inputs);
        for (i, &(a, b, m)) in inputs.iter().enumerate() {
            let cpu_out = f_cpu(a, b, m);
            assert_eq!(
                gpu_out[i], cpu_out,
                "{name} mismatch at i={i}: a={a:016x} b={b:016x} mode={m:?}\n  gpu {:016x}\n  cpu {:016x}",
                gpu_out[i], cpu_out,
            );
        }
    }

    #[test]
    fn gpu_fadd_matches_cpu() {
        check_op2("fadd", cpu::fadd, fadd_batch, 0x1111_1111);
    }
    #[test]
    fn gpu_fsub_matches_cpu() {
        check_op2("fsub", cpu::fsub, fsub_batch, 0x2222_2222);
    }
    #[test]
    fn gpu_fmul_matches_cpu() {
        check_op2("fmul", cpu::fmul, fmul_batch, 0x3333_3333);
    }
    #[test]
    fn gpu_fdiv_matches_cpu() {
        check_op2("fdiv", cpu::fdiv, fdiv_batch, 0x4444_4444);
    }
    #[test]
    fn gpu_fsqrt_matches_cpu() {
        let inputs = gen_op1(0x5555_5555, 1024, true);
        let gpu_out = fsqrt_batch(&inputs);
        for (i, &(a, m)) in inputs.iter().enumerate() {
            let cpu_out = cpu::fsqrt(a, m);
            assert_eq!(
                gpu_out[i], cpu_out,
                "fsqrt mismatch at i={i}: a={a:016x} mode={m:?}\n  gpu {:016x}\n  cpu {:016x}",
                gpu_out[i], cpu_out,
            );
        }
    }

    // --- Full-domain GPU ↔ CPU cross-check --------------------------------
    // Sample any u64 (NaN / ±Inf / ±0 / subnormal / normal) and confirm the
    // GPU MSL kernels match the IEEE-754-conformant softfloat_ref CPU path.
    // NaN bit patterns are not compared verbatim — both sides may pick
    // different payloads but every NaN-output is class-equivalent.

    use crate::softfloat_ref;

    fn class_eq_or_bits(gpu: u64, cpu: u64) -> bool {
        let g_class = softfloat_ref::classify(gpu);
        let c_class = softfloat_ref::classify(cpu);
        if g_class == softfloat_ref::IeeeClass::NaN && c_class == softfloat_ref::IeeeClass::NaN {
            return true;
        }
        gpu == cpu
    }

    fn gen_full_domain_op2(seed: u64, n: usize) -> Vec<(u64, u64, RoundingMode)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let modes = [
            RoundingMode::Nearest,
            RoundingMode::Down,
            RoundingMode::Up,
            RoundingMode::Zero,
        ];
        (0..n)
            .map(|i| {
                (rand_any_bits(&mut rng), rand_any_bits(&mut rng), modes[i % 4])
            })
            .collect()
    }

    fn gen_full_domain_op1(seed: u64, n: usize) -> Vec<(u64, RoundingMode)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let modes = [
            RoundingMode::Nearest,
            RoundingMode::Down,
            RoundingMode::Up,
            RoundingMode::Zero,
        ];
        (0..n).map(|i| (rand_any_bits(&mut rng), modes[i % 4])).collect()
    }

    fn rand_any_bits(rng: &mut impl Rng) -> u64 {
        match rng.gen_range(0u32..100) {
            0..=4 => f64::NAN.to_bits() ^ rng.gen::<u64>(),
            5..=9 => 0x7FF0_0000_0000_0000 | ((rng.gen::<u64>() & 1) << 63),
            10..=14 => (rng.gen::<u64>() & 1) << 63,
            15..=29 => {
                let mant = (rng.gen::<u64>() & 0x000F_FFFF_FFFF_FFFF).max(1);
                ((rng.gen::<u64>() & 1) << 63) | mant
            }
            _ => rand_normal(rng).to_bits(),
        }
    }

    #[test]
    fn gpu_fadd_full_domain() {
        let inputs = gen_full_domain_op2(0xa1_de_b00b, 2048);
        let gpu_out = fadd_batch(&inputs);
        let mut misses = 0u32;
        for (i, &(a, b, mode)) in inputs.iter().enumerate() {
            let want = softfloat_ref::fadd(a, b, mode);
            if !class_eq_or_bits(gpu_out[i], want) {
                if misses < 5 {
                    eprintln!("fadd[{i}] a={a:016x} b={b:016x} mode={mode:?} gpu={:016x} cpu={want:016x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "fadd: {misses} GPU↔CPU mismatches");
    }

    #[test]
    fn gpu_fsub_full_domain() {
        let inputs = gen_full_domain_op2(0xa2_de_b00b, 2048);
        let gpu_out = fsub_batch(&inputs);
        let mut misses = 0u32;
        for (i, &(a, b, mode)) in inputs.iter().enumerate() {
            let want = softfloat_ref::fsub(a, b, mode);
            if !class_eq_or_bits(gpu_out[i], want) {
                if misses < 5 {
                    eprintln!("fsub[{i}] a={a:016x} b={b:016x} mode={mode:?} gpu={:016x} cpu={want:016x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "fsub: {misses} GPU↔CPU mismatches");
    }

    #[test]
    fn gpu_fmul_full_domain() {
        let inputs = gen_full_domain_op2(0xa3_de_b00b, 2048);
        let gpu_out = fmul_batch(&inputs);
        let mut misses = 0u32;
        for (i, &(a, b, mode)) in inputs.iter().enumerate() {
            let want = softfloat_ref::fmul(a, b, mode);
            if !class_eq_or_bits(gpu_out[i], want) {
                if misses < 5 {
                    eprintln!("fmul[{i}] a={a:016x} b={b:016x} mode={mode:?} gpu={:016x} cpu={want:016x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "fmul: {misses} GPU↔CPU mismatches");
    }

    #[test]
    fn gpu_fdiv_full_domain() {
        let inputs = gen_full_domain_op2(0xa4_de_b00b, 2048);
        let gpu_out = fdiv_batch(&inputs);
        let mut misses = 0u32;
        for (i, &(a, b, mode)) in inputs.iter().enumerate() {
            let want = softfloat_ref::fdiv(a, b, mode);
            if !class_eq_or_bits(gpu_out[i], want) {
                if misses < 5 {
                    eprintln!("fdiv[{i}] a={a:016x} b={b:016x} mode={mode:?} gpu={:016x} cpu={want:016x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "fdiv: {misses} GPU↔CPU mismatches");
    }

    #[test]
    fn gpu_fma_full_domain() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xa6_de_b00b);
        let modes = [
            RoundingMode::Nearest,
            RoundingMode::Down,
            RoundingMode::Up,
            RoundingMode::Zero,
        ];
        let triples: Vec<(u64, u64, u64, RoundingMode)> = (0..2048)
            .map(|i| {
                (
                    rand_any_bits(&mut rng),
                    rand_any_bits(&mut rng),
                    rand_any_bits(&mut rng),
                    modes[i % 4],
                )
            })
            .collect();
        let gpu_out = fma_batch(&triples);
        let mut misses = 0u32;
        for (i, &(a, b, c, mode)) in triples.iter().enumerate() {
            let want = softfloat_ref::fma(a, b, c, mode);
            if !class_eq_or_bits(gpu_out[i], want) {
                if misses < 5 {
                    eprintln!("fma[{i}] a={a:016x} b={b:016x} c={c:016x} mode={mode:?} gpu={:016x} cpu={want:016x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "fma: {misses} GPU↔CPU mismatches");
    }

    // Forces fdiv into the subnormal-output (gradual-underflow) branch and
    // confirms the GPU result matches the CPU softfloat_ref bit-for-bit.
    // 70 cases × 4 modes — every input pair has true quotient < 2^-1022, so
    // the round_and_pack subnormal path is the one being measured here.
    #[test]
    fn gpu_fdiv_subnormal_output_matches_cpu() {
        let modes = [
            RoundingMode::Nearest,
            RoundingMode::Down,
            RoundingMode::Up,
            RoundingMode::Zero,
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xa7_de_b00b);
        let mut inputs: Vec<(u64, u64, RoundingMode)> = Vec::new();
        for i in 0..70 {
            // Tiny / huge → result deep into the subnormal range.
            let a_exp_unbiased: i64 = -1000 - (rng.gen_range(0..50) as i64);
            let b_exp_unbiased: i64 = 800 + (rng.gen_range(0..200) as i64);
            let a_exp_biased = (a_exp_unbiased + 1023).max(1) as u64;
            let b_exp_biased = (b_exp_unbiased + 1023).min(2046) as u64;
            let a_mant = rng.gen::<u64>() & 0x000F_FFFF_FFFF_FFFF;
            let b_mant = rng.gen::<u64>() & 0x000F_FFFF_FFFF_FFFF;
            let a_sign = (rng.gen::<u64>() & 1) << 63;
            let b_sign = (rng.gen::<u64>() & 1) << 63;
            let a = a_sign | (a_exp_biased << 52) | a_mant;
            let b = b_sign | (b_exp_biased << 52) | b_mant;
            for mode in modes {
                let _ = i;
                inputs.push((a, b, mode));
            }
        }
        let gpu_out = fdiv_batch(&inputs);
        let mut subnormal_seen = 0u32;
        let mut misses = 0u32;
        for (i, &(a, b, mode)) in inputs.iter().enumerate() {
            let want = softfloat_ref::fdiv(a, b, mode);
            if softfloat_ref::classify(want) == softfloat_ref::IeeeClass::Subnormal {
                subnormal_seen += 1;
            }
            if !class_eq_or_bits(gpu_out[i], want) {
                if misses < 5 {
                    eprintln!("fdiv[{i}] a={a:016x} b={b:016x} mode={mode:?} gpu={:016x} cpu={want:016x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "fdiv subnormal: {misses} GPU↔CPU mismatches");
        assert!(
            subnormal_seen > 0,
            "test inputs did not actually exercise the subnormal-output path"
        );
    }

    #[test]
    fn gpu_fsqrt_full_domain() {
        let inputs = gen_full_domain_op1(0xa5_de_b00b, 2048);
        let gpu_out = fsqrt_batch(&inputs);
        let mut misses = 0u32;
        for (i, &(a, mode)) in inputs.iter().enumerate() {
            let want = softfloat_ref::fsqrt(a, mode);
            if !class_eq_or_bits(gpu_out[i], want) {
                if misses < 5 {
                    eprintln!("fsqrt[{i}] a={a:016x} mode={mode:?} gpu={:016x} cpu={want:016x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "fsqrt: {misses} GPU↔CPU mismatches");
    }

    // --- Conversions: GPU ↔ softfloat_ref ---------------------------------
    //
    // Inputs span every interesting integer / float corner: small values
    // near zero, mantissa-cliff values just below / above 2^k, the i64
    // saturation boundary at ±2^63, and the f64 ↔ f32 overflow / subnormal
    // edges. Each input is tested across all four rounding modes.

    const MODES4: &[RoundingMode] = &[
        RoundingMode::Nearest,
        RoundingMode::Down,
        RoundingMode::Up,
        RoundingMode::Zero,
    ];

    fn gen_i64_inputs(seed: u64, n: usize) -> Vec<(i64, RoundingMode)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut v: Vec<(i64, RoundingMode)> = Vec::new();
        // Hand-picked corners: zero, ±1, saturation boundary, mantissa cliffs.
        let corners: &[i64] = &[
            0,
            1,
            -1,
            i64::MIN,
            i64::MAX,
            i64::MIN + 1,
            i64::MAX - 1,
            (1i64 << 53) - 1,
            1i64 << 53,
            (1i64 << 53) + 1,
            -(1i64 << 53),
            -((1i64 << 53) + 1),
            (1i64 << 62) + 0x123,
        ];
        for &c in corners {
            for &m in MODES4 {
                v.push((c, m));
            }
        }
        for i in 0..n {
            // Bias the random distribution toward big magnitudes so the
            // round-to-f64 path actually drops bits.
            let shift: u32 = rng.gen_range(0..63);
            let raw: i64 = rng.gen::<i64>() >> shift;
            v.push((raw, MODES4[i % 4]));
        }
        v
    }

    #[test]
    fn gpu_cvt_i64_to_f64_matches_cpu() {
        let inputs = gen_i64_inputs(0xc1_de_b00b, 1024);
        let gpu_out = cvt_i64_to_f64_batch(&inputs);
        let mut misses = 0u32;
        for (i, &(x, m)) in inputs.iter().enumerate() {
            let want = softfloat_ref::cvt_i64_to_f64(x, m);
            if gpu_out[i] != want {
                if misses < 5 {
                    eprintln!("cvt_i64_to_f64[{i}] x={x} mode={m:?} gpu={:016x} cpu={want:016x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "cvt_i64_to_f64: {misses} mismatches");
    }

    #[test]
    fn gpu_cvt_u64_to_f64_matches_cpu() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc2_de_b00b);
        let mut inputs: Vec<(u64, RoundingMode)> = Vec::new();
        let corners: &[u64] = &[
            0, 1, u64::MAX, u64::MAX - 1,
            (1u64 << 53) - 1, 1u64 << 53, (1u64 << 53) + 1,
            1u64 << 63, (1u64 << 63) + 1,
        ];
        for &c in corners { for &m in MODES4 { inputs.push((c, m)); } }
        for i in 0..1024 {
            let shift: u32 = rng.gen_range(0..63);
            let raw: u64 = rng.gen::<u64>() >> shift;
            inputs.push((raw, MODES4[i % 4]));
        }
        let gpu_out = cvt_u64_to_f64_batch(&inputs);
        let mut misses = 0u32;
        for (i, &(x, m)) in inputs.iter().enumerate() {
            let want = softfloat_ref::cvt_u64_to_f64(x, m);
            if gpu_out[i] != want {
                if misses < 5 {
                    eprintln!("cvt_u64_to_f64[{i}] x={x} mode={m:?} gpu={:016x} cpu={want:016x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "cvt_u64_to_f64: {misses} mismatches");
    }

    #[test]
    fn gpu_cvt_f64_to_i64_matches_cpu() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc3_de_b00b);
        let mut inputs: Vec<(u64, RoundingMode)> = Vec::new();
        // Special operands and round-half / saturation corners.
        let corners: &[f64] = &[
            0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5,
            (i64::MAX as f64), (i64::MIN as f64),
            (i64::MAX as f64) * 2.0, (i64::MIN as f64) * 2.0,
            f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
            f64::from_bits(1), f64::from_bits(0x800F_FFFF_FFFF_FFFF), // ±tiny subnormal
        ];
        for &c in corners {
            for &m in MODES4 {
                inputs.push((c.to_bits(), m));
            }
        }
        for i in 0..1024 {
            // Cover the full integer range plus fractional / out-of-range.
            let shift: i32 = rng.gen_range(-3..=66);
            let mant: f64 = rng.gen::<f64>() * 2.0 - 1.0;
            let val = mant * 2f64.powi(shift);
            inputs.push((val.to_bits(), MODES4[i % 4]));
        }
        let gpu_out = cvt_f64_to_i64_batch(&inputs);
        let mut misses = 0u32;
        for (i, &(a, m)) in inputs.iter().enumerate() {
            let want = softfloat_ref::cvt_f64_to_i64(a, m);
            if gpu_out[i] != want {
                if misses < 5 {
                    eprintln!("cvt_f64_to_i64[{i}] a={a:016x} mode={m:?} gpu={} cpu={want}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "cvt_f64_to_i64: {misses} mismatches");
    }

    #[test]
    fn gpu_cvt_f32_to_f64_matches_cpu() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc4_de_b00b);
        let mut inputs: Vec<u32> = Vec::new();
        // Specials + subnormal-cliff and small-mantissa corners.
        let corners: &[u32] = &[
            0u32, 0x8000_0000,                              // ±0
            0x7F80_0000, 0xFF80_0000,                       // ±Inf
            0x7FC0_0000, 0xFFC0_0000,                       // qNaN
            0x7FA0_0000,                                    // sNaN
            0x0000_0001, 0x8000_0001,                       // smallest subnormal
            0x007F_FFFF, 0x807F_FFFF,                       // largest subnormal
            0x0080_0000, 0x8080_0000,                       // smallest normal
            0x7F7F_FFFF, 0xFF7F_FFFF,                       // largest finite
            0x3F80_0000,                                    // 1.0
        ];
        inputs.extend_from_slice(corners);
        for _ in 0..1024 {
            inputs.push(rng.gen::<u32>());
        }
        let gpu_out = cvt_f32_to_f64_batch(&inputs);
        let mut misses = 0u32;
        for (i, &x) in inputs.iter().enumerate() {
            let want = softfloat_ref::cvt_f32_to_f64(x);
            if !class_eq_or_bits(gpu_out[i], want) {
                if misses < 5 {
                    eprintln!("cvt_f32_to_f64[{i}] x={x:08x} gpu={:016x} cpu={want:016x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "cvt_f32_to_f64: {misses} mismatches");
    }

    #[test]
    fn gpu_cvt_f64_to_f32_matches_cpu() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc5_de_b00b);
        let mut inputs: Vec<(u64, RoundingMode)> = Vec::new();
        let corners: &[f64] = &[
            0.0, -0.0,
            f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
            1.0, -1.0,
            f64::from_bits(0x47EF_FFFF_E000_0000), // f32::MAX as f64
            f64::from_bits(0x47EF_FFFF_F000_0000), // exactly halfway between f32::MAX and f32 inf
            f64::from_bits(0x4800_0000_0000_0000), // > f32::MAX → overflow
            f64::from_bits(0x3690_0000_0000_0000), // 2^-149 = smallest f32 subnormal
            f64::from_bits(0x3680_0000_0000_0000), // 2^-150 = below f32 subnormal
            f64::from_bits(0x36A0_0000_0000_0000), // 2^-148 = 2x smallest f32 subnormal
        ];
        for &c in corners {
            for &m in MODES4 {
                inputs.push((c.to_bits(), m));
            }
        }
        for i in 0..1024 {
            // Sample across the full f64 exponent range so we hit f32
            // overflow, f32 normal, f32 subnormal, and underflow paths.
            let exp: i32 = rng.gen_range(-160..=130);
            let mant: f64 = rng.gen::<f64>() * 2.0 - 1.0;
            let val = mant * 2f64.powi(exp);
            inputs.push((val.to_bits(), MODES4[i % 4]));
        }
        let gpu_out = cvt_f64_to_f32_batch(&inputs);
        let mut misses = 0u32;
        for (i, &(a, m)) in inputs.iter().enumerate() {
            let want = softfloat_ref::cvt_f64_to_f32(a, m);
            // NaN payloads may differ; check class equivalence for f32.
            let is_nan_gpu = ((gpu_out[i] >> 23) & 0xFF == 0xFF) && (gpu_out[i] & 0x7F_FFFF) != 0;
            let is_nan_cpu = ((want >> 23) & 0xFF == 0xFF) && (want & 0x7F_FFFF) != 0;
            let ok = (is_nan_gpu && is_nan_cpu) || gpu_out[i] == want;
            if !ok {
                if misses < 5 {
                    eprintln!("cvt_f64_to_f32[{i}] a={a:016x} mode={m:?} gpu={:08x} cpu={want:08x}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "cvt_f64_to_f32: {misses} mismatches");
    }

    // --- Comparisons: GPU ↔ softfloat_ref ---------------------------------
    //
    // Comparisons are mode-independent; we just need broad coverage of the
    // sign / class space, especially ±0 and any-NaN.

    fn gen_cmp_inputs(seed: u64, n: usize) -> Vec<(u64, u64)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let corners: &[u64] = &[
            0, 1u64 << 63,                                     // ±0
            0x3FF0_0000_0000_0000, 0xBFF0_0000_0000_0000,      // ±1.0
            0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000,      // ±Inf
            0x7FF8_0000_0000_0000, 0xFFF8_0000_0000_0000,      // qNaN
            0x7FF4_0000_0000_0000,                             // sNaN
            1, 1u64 << 52,                                     // tiny subnormal / smallest normal
        ];
        let mut v: Vec<(u64, u64)> = Vec::new();
        for &a in corners {
            for &b in corners {
                v.push((a, b));
            }
        }
        for _ in 0..n {
            v.push((rand_any_bits(&mut rng), rand_any_bits(&mut rng)));
        }
        // Sprinkle in some equal pairs so the eq path gets a true outcome.
        for &c in corners {
            v.push((c, c));
        }
        v
    }

    fn check_cmp(name: &str, cpu: fn(u64, u64) -> bool, gpu: fn(&[(u64, u64)]) -> Vec<bool>, seed: u64) {
        let inputs = gen_cmp_inputs(seed, 1024);
        let gpu_out = gpu(&inputs);
        let mut misses = 0u32;
        for (i, &(a, b)) in inputs.iter().enumerate() {
            let want = cpu(a, b);
            if gpu_out[i] != want {
                if misses < 5 {
                    eprintln!("{name}[{i}] a={a:016x} b={b:016x} gpu={} cpu={want}",
                        gpu_out[i]);
                }
                misses += 1;
            }
        }
        assert_eq!(misses, 0, "{name}: {misses} mismatches");
    }

    #[test]
    fn gpu_feq_matches_cpu() { check_cmp("feq", softfloat_ref::feq, feq_batch, 0xe1_de_b00b); }
    #[test]
    fn gpu_flt_matches_cpu() { check_cmp("flt", softfloat_ref::flt, flt_batch, 0xe2_de_b00b); }
    #[test]
    fn gpu_fle_matches_cpu() { check_cmp("fle", softfloat_ref::fle, fle_batch, 0xe3_de_b00b); }
    #[test]
    fn gpu_fgt_matches_cpu() { check_cmp("fgt", softfloat_ref::fgt, fgt_batch, 0xe4_de_b00b); }
    #[test]
    fn gpu_fge_matches_cpu() { check_cmp("fge", softfloat_ref::fge, fge_batch, 0xe5_de_b00b); }
}
