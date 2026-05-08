//! Per-op throughput benchmarks for the three backends:
//!
//! - **GPU softfloat** — Metal kernel, one thread per input.
//! - **CPU softfloat_ref** — single-threaded integer-arithmetic reference
//!   (same algorithm as the GPU kernel, but on one CPU core).
//! - **CPU native f64** — single-threaded hardware FPU.
//!
//! Run:
//!
//! ```sh
//! cargo bench -p metal-softfloat --features gpu --bench op_throughput
//! ```
//!
//! Criterion writes per-op timing summaries plus an HTML report under
//! `target/criterion/`. To filter to one op, append `-- fadd` (etc.).
//!
//! macOS / Apple Silicon only — Metal lives there. The bench has
//! `required-features = ["gpu"]`, so a default `cargo bench` skips it.
//!
//! Notes:
//!
//! - The GPU backend is batch-only (one-shot launch + readback), so the
//!   benchmark dispatches a fixed-size batch and reports total time per
//!   batch. Per-op throughput is `BATCH_SIZE / time`.
//! - Inputs are sampled once via `rand::StdRng` with a fixed seed so
//!   measurements are reproducible across runs.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use metal_softfloat::{gpu, softfloat_ref, RoundingMode};
use rand::{Rng, SeedableRng};

// Native-f64 reference ops, used as the "what the hardware FPU costs"
// baseline. Hardcoded to nearest-rounding because the bench feeds only
// `RoundingMode::Nearest` — we don't need (or want) the FPCR-mutating
// `cpu::*` machinery here.
fn native_fadd(a: u64, b: u64, _m: RoundingMode) -> u64 { (f64::from_bits(a) + f64::from_bits(b)).to_bits() }
fn native_fsub(a: u64, b: u64, _m: RoundingMode) -> u64 { (f64::from_bits(a) - f64::from_bits(b)).to_bits() }
fn native_fmul(a: u64, b: u64, _m: RoundingMode) -> u64 { (f64::from_bits(a) * f64::from_bits(b)).to_bits() }
fn native_fdiv(a: u64, b: u64, _m: RoundingMode) -> u64 { (f64::from_bits(a) / f64::from_bits(b)).to_bits() }
fn native_fsqrt(a: u64, _m: RoundingMode) -> u64 { f64::from_bits(a).sqrt().to_bits() }
fn native_fma(a: u64, b: u64, c: u64, _m: RoundingMode) -> u64 {
    f64::from_bits(a).mul_add(f64::from_bits(b), f64::from_bits(c)).to_bits()
}

const BATCH_SIZE: usize = 1 << 20; // ~1M ops per dispatch

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

fn gen_op2(seed: u64) -> Vec<(u64, u64, RoundingMode)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..BATCH_SIZE)
        .map(|_| {
            (
                rand_normal(&mut rng).to_bits(),
                rand_normal(&mut rng).to_bits(),
                RoundingMode::Nearest,
            )
        })
        .collect()
}

fn gen_op1_pos(seed: u64) -> Vec<(u64, RoundingMode)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..BATCH_SIZE)
        .map(|_| (rand_normal(&mut rng).abs().to_bits(), RoundingMode::Nearest))
        .collect()
}

fn gen_op3(seed: u64) -> Vec<(u64, u64, u64, RoundingMode)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..BATCH_SIZE)
        .map(|_| {
            (
                rand_normal(&mut rng).to_bits(),
                rand_normal(&mut rng).to_bits(),
                rand_normal(&mut rng).to_bits(),
                RoundingMode::Nearest,
            )
        })
        .collect()
}

fn bench_op2(
    c: &mut Criterion,
    name: &str,
    gpu_batch: fn(&[(u64, u64, RoundingMode)]) -> Vec<u64>,
    softfloat_ref_op: fn(u64, u64, RoundingMode) -> u64,
    native_op: fn(u64, u64, RoundingMode) -> u64,
) {
    let inputs = gen_op2(0xdead_beef ^ name.len() as u64);
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("gpu_softfloat", |b| {
        b.iter(|| {
            let out = gpu_batch(black_box(&inputs));
            black_box(out);
        });
    });

    group.bench_function("cpu_softfloat_ref", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(a, b_, m) in &inputs {
                acc ^= softfloat_ref_op(black_box(a), black_box(b_), m);
            }
            black_box(acc);
        });
    });

    group.bench_function("cpu_native_f64", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(a, b_, m) in &inputs {
                acc ^= native_op(black_box(a), black_box(b_), m);
            }
            black_box(acc);
        });
    });

    group.finish();
}

fn bench_op1_pos(
    c: &mut Criterion,
    name: &str,
    gpu_batch: fn(&[(u64, RoundingMode)]) -> Vec<u64>,
    softfloat_ref_op: fn(u64, RoundingMode) -> u64,
    native_op: fn(u64, RoundingMode) -> u64,
) {
    let inputs = gen_op1_pos(0xcafe_d00d ^ name.len() as u64);
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("gpu_softfloat", |b| {
        b.iter(|| {
            let out = gpu_batch(black_box(&inputs));
            black_box(out);
        });
    });

    group.bench_function("cpu_softfloat_ref", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(a, m) in &inputs {
                acc ^= softfloat_ref_op(black_box(a), m);
            }
            black_box(acc);
        });
    });

    group.bench_function("cpu_native_f64", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(a, m) in &inputs {
                acc ^= native_op(black_box(a), m);
            }
            black_box(acc);
        });
    });

    group.finish();
}

fn bench_op3(
    c: &mut Criterion,
    name: &str,
    gpu_batch: fn(&[(u64, u64, u64, RoundingMode)]) -> Vec<u64>,
    softfloat_ref_op: fn(u64, u64, u64, RoundingMode) -> u64,
    native_op: fn(u64, u64, u64, RoundingMode) -> u64,
) {
    let inputs = gen_op3(0xfeed_face ^ name.len() as u64);
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("gpu_softfloat", |b| {
        b.iter(|| {
            let out = gpu_batch(black_box(&inputs));
            black_box(out);
        });
    });

    group.bench_function("cpu_softfloat_ref", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(a, b_, c_, m) in &inputs {
                acc ^= softfloat_ref_op(black_box(a), black_box(b_), black_box(c_), m);
            }
            black_box(acc);
        });
    });

    group.bench_function("cpu_native_f64", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(a, b_, c_, m) in &inputs {
                acc ^= native_op(black_box(a), black_box(b_), black_box(c_), m);
            }
            black_box(acc);
        });
    });

    group.finish();
}

// =============================================================================
// Conversion / comparison benches.
//
// Each new op has a slightly different signature shape (i64/u64/u32/bool
// crossing the boundary), so they don't fit `bench_op1`/`op2`/`op3`. The
// per-op blocks below dispatch GPU / softfloat_ref / native f64 against
// the same inputs so the three columns are directly comparable, matching
// the structure of the arithmetic benches above.
// =============================================================================

fn gen_i64(seed: u64) -> Vec<(i64, RoundingMode)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..BATCH_SIZE)
        .map(|_| {
            // Bias toward big magnitudes so the round-to-f64 path actually
            // drops bits (small i64s convert exactly).
            let shift: u32 = rng.gen_range(0..63);
            (rng.gen::<i64>() >> shift, RoundingMode::Nearest)
        })
        .collect()
}

fn gen_u64(seed: u64) -> Vec<(u64, RoundingMode)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..BATCH_SIZE)
        .map(|_| {
            let shift: u32 = rng.gen_range(0..63);
            (rng.gen::<u64>() >> shift, RoundingMode::Nearest)
        })
        .collect()
}

fn gen_f64_for_to_int(seed: u64) -> Vec<(u64, RoundingMode)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..BATCH_SIZE)
        .map(|_| {
            // Sample across the full integer range plus some out-of-range
            // values so the saturation path gets some traffic.
            let exp: i32 = rng.gen_range(-3..=66);
            let mant: f64 = rng.gen::<f64>() * 2.0 - 1.0;
            (mant * 2f64.powi(exp), RoundingMode::Nearest)
        })
        .map(|(v, m)| (v.to_bits(), m))
        .collect()
}

fn gen_f32(seed: u64) -> Vec<u32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..BATCH_SIZE).map(|_| rng.gen::<u32>()).collect()
}

fn bench_cvt_i64_to_f64(c: &mut Criterion) {
    let inputs = gen_i64(0xc1_c1_c1_c1);
    let mut group = c.benchmark_group("cvt_i64_to_f64");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("gpu_softfloat", |b| {
        b.iter(|| black_box(gpu::cvt_i64_to_f64_batch(black_box(&inputs))));
    });
    group.bench_function("cpu_softfloat_ref", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(x, m) in &inputs {
                acc ^= softfloat_ref::cvt_i64_to_f64(black_box(x), m);
            }
            black_box(acc);
        });
    });
    group.bench_function("cpu_native_f64", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(x, _m) in &inputs {
                acc ^= (black_box(x) as f64).to_bits();
            }
            black_box(acc);
        });
    });
    group.finish();
}

fn bench_cvt_u64_to_f64(c: &mut Criterion) {
    let inputs = gen_u64(0xc2_c2_c2_c2);
    let mut group = c.benchmark_group("cvt_u64_to_f64");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("gpu_softfloat", |b| {
        b.iter(|| black_box(gpu::cvt_u64_to_f64_batch(black_box(&inputs))));
    });
    group.bench_function("cpu_softfloat_ref", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(x, m) in &inputs {
                acc ^= softfloat_ref::cvt_u64_to_f64(black_box(x), m);
            }
            black_box(acc);
        });
    });
    group.bench_function("cpu_native_f64", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(x, _m) in &inputs {
                acc ^= (black_box(x) as f64).to_bits();
            }
            black_box(acc);
        });
    });
    group.finish();
}

fn bench_cvt_f64_to_i64(c: &mut Criterion) {
    let inputs = gen_f64_for_to_int(0xc3_c3_c3_c3);
    let mut group = c.benchmark_group("cvt_f64_to_i64");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("gpu_softfloat", |b| {
        b.iter(|| black_box(gpu::cvt_f64_to_i64_batch(black_box(&inputs))));
    });
    group.bench_function("cpu_softfloat_ref", |b| {
        b.iter(|| {
            let mut acc = 0i64;
            for &(a, m) in &inputs {
                acc ^= softfloat_ref::cvt_f64_to_i64(black_box(a), m);
            }
            black_box(acc);
        });
    });
    group.bench_function("cpu_native_f64", |b| {
        b.iter(|| {
            let mut acc = 0i64;
            for &(a, _m) in &inputs {
                // Rust `as i64` from f64 saturates and maps NaN→0, matching
                // softfloat_ref::cvt_f64_to_i64's semantics on those inputs.
                acc ^= f64::from_bits(black_box(a)) as i64;
            }
            black_box(acc);
        });
    });
    group.finish();
}

fn bench_cvt_f32_to_f64(c: &mut Criterion) {
    let inputs = gen_f32(0xc4_c4_c4_c4);
    let mut group = c.benchmark_group("cvt_f32_to_f64");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("gpu_softfloat", |b| {
        b.iter(|| black_box(gpu::cvt_f32_to_f64_batch(black_box(&inputs))));
    });
    group.bench_function("cpu_softfloat_ref", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &x in &inputs {
                acc ^= softfloat_ref::cvt_f32_to_f64(black_box(x));
            }
            black_box(acc);
        });
    });
    group.bench_function("cpu_native_f64", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &x in &inputs {
                acc ^= (f32::from_bits(black_box(x)) as f64).to_bits();
            }
            black_box(acc);
        });
    });
    group.finish();
}

fn bench_cvt_f64_to_f32(c: &mut Criterion) {
    let inputs = gen_op2(0xc5_c5_c5_c5)
        .into_iter()
        .map(|(a, _b, m)| (a, m))
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("cvt_f64_to_f32");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("gpu_softfloat", |b| {
        b.iter(|| black_box(gpu::cvt_f64_to_f32_batch(black_box(&inputs))));
    });
    group.bench_function("cpu_softfloat_ref", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &(a, m) in &inputs {
                acc ^= softfloat_ref::cvt_f64_to_f32(black_box(a), m);
            }
            black_box(acc);
        });
    });
    group.bench_function("cpu_native_f64", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &(a, _m) in &inputs {
                acc ^= (f64::from_bits(black_box(a)) as f32).to_bits();
            }
            black_box(acc);
        });
    });
    group.finish();
}

fn bench_cmp(
    c: &mut Criterion,
    name: &str,
    gpu_batch: fn(&[(u64, u64)]) -> Vec<bool>,
    softfloat_ref_op: fn(u64, u64) -> bool,
    native_op: fn(u64, u64) -> bool,
) {
    // Comparisons ignore the rounding mode, so reuse the op2 generator
    // and drop the mode field for the GPU batch input.
    let inputs_full = gen_op2(0xe0_de_b00b ^ name.len() as u64);
    let inputs: Vec<(u64, u64)> = inputs_full.iter().map(|&(a, b, _)| (a, b)).collect();
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("gpu_softfloat", |b| {
        b.iter(|| black_box(gpu_batch(black_box(&inputs))));
    });
    group.bench_function("cpu_softfloat_ref", |b| {
        b.iter(|| {
            let mut acc: u32 = 0;
            for &(a, b_) in &inputs {
                acc = acc.wrapping_add(u32::from(softfloat_ref_op(black_box(a), black_box(b_))));
            }
            black_box(acc);
        });
    });
    group.bench_function("cpu_native_f64", |b| {
        b.iter(|| {
            let mut acc: u32 = 0;
            for &(a, b_) in &inputs {
                acc = acc.wrapping_add(u32::from(native_op(black_box(a), black_box(b_))));
            }
            black_box(acc);
        });
    });
    group.finish();
}

// IEEE-754 quiet semantics on native f64: any-NaN comparison is false.
// `==` and `<` etc. on `f64` already implement that.
fn native_feq(a: u64, b: u64) -> bool { f64::from_bits(a) == f64::from_bits(b) }
fn native_flt(a: u64, b: u64) -> bool { f64::from_bits(a) <  f64::from_bits(b) }
fn native_fle(a: u64, b: u64) -> bool { f64::from_bits(a) <= f64::from_bits(b) }
fn native_fgt(a: u64, b: u64) -> bool { f64::from_bits(a) >  f64::from_bits(b) }
fn native_fge(a: u64, b: u64) -> bool { f64::from_bits(a) >= f64::from_bits(b) }

fn benches(c: &mut Criterion) {
    bench_op2(c, "fadd", gpu::fadd_batch, softfloat_ref::fadd, native_fadd);
    bench_op2(c, "fsub", gpu::fsub_batch, softfloat_ref::fsub, native_fsub);
    bench_op2(c, "fmul", gpu::fmul_batch, softfloat_ref::fmul, native_fmul);
    bench_op2(c, "fdiv", gpu::fdiv_batch, softfloat_ref::fdiv, native_fdiv);
    bench_op1_pos(c, "fsqrt", gpu::fsqrt_batch, softfloat_ref::fsqrt, native_fsqrt);
    bench_op3(c, "fma", gpu::fma_batch, softfloat_ref::fma, native_fma);

    bench_cvt_i64_to_f64(c);
    bench_cvt_u64_to_f64(c);
    bench_cvt_f64_to_i64(c);
    bench_cvt_f32_to_f64(c);
    bench_cvt_f64_to_f32(c);

    bench_cmp(c, "feq", gpu::feq_batch, softfloat_ref::feq, native_feq);
    bench_cmp(c, "flt", gpu::flt_batch, softfloat_ref::flt, native_flt);
    bench_cmp(c, "fle", gpu::fle_batch, softfloat_ref::fle, native_fle);
    bench_cmp(c, "fgt", gpu::fgt_batch, softfloat_ref::fgt, native_fgt);
    bench_cmp(c, "fge", gpu::fge_batch, softfloat_ref::fge, native_fge);

    chain_benches(c);
}

// =============================================================================
// Chain-kernel throughput.
//
// The dispatch-bound `*_batch` benches above measure host↔device latency
// + kernel launch overhead. They're useful for sanity but flat at ~3 ms
// for almost every op because dispatch dominates. The chain kernels pin
// ~1024 ops on the GPU per thread before any host I/O, so what gets
// measured is the actual softfloat throughput. These are the numbers
// worth tracking over time.
//
// Each chain bench reports two columns: the GPU chain kernel and a
// single-threaded CPU native-f64 baseline running the same op count
// with the same data-dependency shape (so neither side gets to ILP-
// parallelise across iterations). softfloat_ref isn't shown here — its
// per-op numbers come from the `*_batch` rows above.
// =============================================================================

// Tuned for a balance between GPU saturation (more lanes = closer to
// peak throughput) and CPU baseline runtime (single-thread chains have
// no ILP, so they scale linearly). 10k × 1024 = ~10 M ops/iter ≈ 30 ms
// CPU on the slow ops, ~3 ms GPU on fast ops. For a one-shot peak
// throughput number on the GPU, use `examples/throughput_demo` with
// THREADS=200000 instead — the criterion bench is for tracking
// regressions, not absolute peak.
const CHAIN_THREADS: usize = 10_000;
const CHAIN_TOTAL_OPS: u64 = (CHAIN_THREADS * gpu::CHAIN_OPS_PER_THREAD) as u64;
const CHAIN_SEED: (u64, u64) = (0x4000_0000_0000_0000, 0x3FF0_0000_0000_0001);

fn cpu_chain_binary(op: impl Fn(u64, u64) -> u64) -> u64 {
    let mut a = 0x3FF0_0000_0000_0000u64;
    let mut b = 0x3FF0_0000_0000_0001u64;
    for _ in 0..CHAIN_TOTAL_OPS {
        a = op(a, b);
        b ^= a & 0xFF;
    }
    a ^ b
}
fn cpu_chain_unary(op: impl Fn(u64) -> u64) -> u64 {
    let mut a = 0x3FF0_0000_0000_0000u64;
    let b = 0x3FF0_0000_0000_0001u64;
    for _ in 0..CHAIN_TOTAL_OPS {
        a = op(a);
        a ^= b & 0xFF;
    }
    a ^ b
}
fn cpu_chain_ternary(op: impl Fn(u64, u64, u64) -> u64) -> u64 {
    let mut a = 0x3FF0_0000_0000_0000u64;
    let mut b = 0x3FF0_0000_0000_0001u64;
    let c_ = 0x3FF0_0000_0000_0000u64;
    for _ in 0..CHAIN_TOTAL_OPS {
        a = op(a, b, c_);
        b ^= a & 0xFF;
    }
    a ^ b
}
fn cpu_chain_cvt(init: u64, op: impl Fn(u64) -> u64) -> u64 {
    let mut x = init;
    let mut acc = 0u64;
    for _ in 0..CHAIN_TOTAL_OPS {
        let r = op(x);
        acc ^= r;
        x ^= r & 0xFF;
    }
    acc ^ x
}
fn cpu_chain_cmp(op: impl Fn(u64, u64) -> u64) -> u64 {
    let a = 0x3FF0_0000_0000_0000u64;
    let mut b = 0x3FF0_0000_0000_0001u64;
    let mut acc = 0u64;
    for _ in 0..CHAIN_TOTAL_OPS {
        // black_box on `b` defeats LLVM constant-propagation through the
        // comparison: without it, the optimizer can prove the result of
        // ops like feq/fgt/fge is invariant (b never crosses a) and
        // delete the whole loop. Measured 380 ps/iter (faster than light
        // crossing a centimetre) before adding the fence.
        let r = op(a, black_box(b));
        acc = (acc << 1) ^ r;
        b ^= (a ^ acc) & 0xFF;
    }
    acc ^ a ^ b
}

fn bench_chain_pair<C: Fn() -> u64>(
    c: &mut Criterion,
    name: &str,
    gpu_chain: fn(usize, (u64, u64)) -> Vec<u64>,
    cpu_chain: C,
) {
    let mut group = c.benchmark_group(format!("chain_{name}"));
    group.throughput(Throughput::Elements(CHAIN_TOTAL_OPS));
    group.bench_function("gpu_softfloat", |b| {
        b.iter(|| black_box(gpu_chain(CHAIN_THREADS, CHAIN_SEED)));
    });
    group.bench_function("cpu_native_f64", |b| {
        b.iter(|| black_box(cpu_chain()));
    });
    group.finish();
}

fn chain_benches(c: &mut Criterion) {
    bench_chain_pair(c, "fadd", gpu::fadd_chain, || cpu_chain_binary(|a, b| native_fadd(a, b, RoundingMode::Nearest)));
    bench_chain_pair(c, "fsub", gpu::fsub_chain, || cpu_chain_binary(|a, b| native_fsub(a, b, RoundingMode::Nearest)));
    bench_chain_pair(c, "fmul", gpu::fmul_chain, || cpu_chain_binary(|a, b| native_fmul(a, b, RoundingMode::Nearest)));
    bench_chain_pair(c, "fdiv", gpu::fdiv_chain, || cpu_chain_binary(|a, b| native_fdiv(a, b, RoundingMode::Nearest)));
    bench_chain_pair(c, "fsqrt", gpu::fsqrt_chain, || cpu_chain_unary(|a| native_fsqrt(a, RoundingMode::Nearest)));
    bench_chain_pair(c, "fma", gpu::fma_chain, || cpu_chain_ternary(|a, b, c_| native_fma(a, b, c_, RoundingMode::Nearest)));

    // Conversion init values mirror the MSL chain seeds: integers near
    // 1000 to avoid drift to saturation, floats near 1.0.
    bench_chain_pair(c, "cvt_i64_to_f64", gpu::cvt_i64_to_f64_chain,
        || cpu_chain_cvt(1000, |x| (x as i64 as f64).to_bits()));
    bench_chain_pair(c, "cvt_u64_to_f64", gpu::cvt_u64_to_f64_chain,
        || cpu_chain_cvt(1000, |x| (x as f64).to_bits()));
    bench_chain_pair(c, "cvt_f64_to_i64", gpu::cvt_f64_to_i64_chain,
        || cpu_chain_cvt(0x3FF0_0000_0000_0000, |a| f64::from_bits(a) as i64 as u64));
    bench_chain_pair(c, "cvt_f32_to_f64", gpu::cvt_f32_to_f64_chain,
        || cpu_chain_cvt(0x3F80_0000, |x| (f32::from_bits(x as u32) as f64).to_bits()));
    bench_chain_pair(c, "cvt_f64_to_f32", gpu::cvt_f64_to_f32_chain,
        || cpu_chain_cvt(0x3FF0_0000_0000_0000, |a| u64::from((f64::from_bits(a) as f32).to_bits())));

    bench_chain_pair(c, "feq", gpu::feq_chain, || cpu_chain_cmp(|a, b| u64::from(f64::from_bits(a) == f64::from_bits(b))));
    bench_chain_pair(c, "flt", gpu::flt_chain, || cpu_chain_cmp(|a, b| u64::from(f64::from_bits(a) <  f64::from_bits(b))));
    bench_chain_pair(c, "fle", gpu::fle_chain, || cpu_chain_cmp(|a, b| u64::from(f64::from_bits(a) <= f64::from_bits(b))));
    bench_chain_pair(c, "fgt", gpu::fgt_chain, || cpu_chain_cmp(|a, b| u64::from(f64::from_bits(a) >  f64::from_bits(b))));
    bench_chain_pair(c, "fge", gpu::fge_chain, || cpu_chain_cmp(|a, b| u64::from(f64::from_bits(a) >= f64::from_bits(b))));
}

// Tighter than criterion defaults (3 s warmup + 5 s measurement, 100
// samples). With 48 sub-benches that adds up to ~7+ minutes of wall
// time. Per-iter cost here is dominated by GPU dispatch overhead and
// 1 M-element loops, both of which are stable enough that 20 samples
// with 1 s measurement give tight intervals — the full suite runs in
// roughly a third of the time without losing useful precision.
fn quick_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(800))
        .measurement_time(Duration::from_secs(2))
        .sample_size(20)
}

criterion_group! {
    name = group;
    config = quick_criterion();
    targets = benches
}
criterion_main!(group);
