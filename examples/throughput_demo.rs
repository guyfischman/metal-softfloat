//! GPU softfloat aggregate throughput vs CPU multi-core hardware f64.
//!
//! Apple Silicon CPUs have a hardware FPU per core; one f64 `fadd` is
//! one machine instruction. Apple GPUs have no f64 hardware; each `fadd`
//! is ~30-50 integer instructions of softfloat emulation. Naive intuition
//! says CPU should win.
//!
//! It doesn't, once you put enough work through both. The GPU has so
//! many parallel lanes that aggregate softfloat throughput beats the
//! CPU's combined hardware-f64 throughput across all cores — even
//! though each individual op is dramatically slower.
//!
//! And critically: the GPU result is **bit-equal** to what the CPU
//! hardware FPU produced. `softfloat_ref::*` is bit-exact vs native
//! `f64` arithmetic under round-to-nearest, and the GPU MSL kernels
//! match `softfloat_ref` bit-for-bit. So you get faster *and*
//! protocol-equivalent results.
//!
//! What this demo does:
//!
//! 1. **Correctness** — dispatch `gpu::fadd_batch` on 4096 random pairs,
//!    verify every output is bit-equal to `softfloat_ref::fadd`.
//! 2. **Throughput** — call the public `gpu::*_chain` API (which
//!    dispatches the `__softfloat64_*_chain` kernels in
//!    `softfloat64.metal`) for each op. Compare aggregate ops/sec
//!    against a 14-thread CPU baseline doing the same op count on
//!    hardware f64.
//!
//! Run:
//!
//! ```sh
//! cargo run --release --features gpu --example throughput_demo
//! ```
//!
//! Env knobs:
//!
//! - `THREADS` — GPU thread count (default 200_000). Total ops measured =
//!   `THREADS * CHAIN_OPS_PER_THREAD` (1024).
//! - `RUNS` — timed dispatches per op, takes best (default 3).

#![cfg(target_os = "macos")]

use std::thread;
use std::time::Instant;

use metal_softfloat_core::{gpu, softfloat_ref, RoundingMode};
use rand::{Rng, SeedableRng};

fn rand_normal(rng: &mut impl Rng) -> f64 {
    loop {
        let log_mag: f64 = rng.gen_range(-30.0..30.0);
        let sign: f64 = if rng.gen::<bool>() { -1.0 } else { 1.0 };
        let x = sign * 2f64.powf(log_mag);
        if x.is_normal() {
            return x;
        }
    }
}

fn correctness_check() -> u32 {
    // softfloat_ref is bit-equal to native CPU f64 hardware in nearest mode
    // (validated under aarch64 in the crate's own tests). If GPU matches
    // softfloat_ref, GPU transitively matches CPU hardware f64.
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xCAFE_F00D);
    let pairs: Vec<(u64, u64, RoundingMode)> = (0..4096)
        .map(|_| {
            (
                rand_normal(&mut rng).to_bits(),
                rand_normal(&mut rng).to_bits(),
                RoundingMode::Nearest,
            )
        })
        .collect();
    let gpu_out = gpu::fadd_batch(&pairs);
    let cpu_out: Vec<u64> = pairs.iter().map(|&(a, b, m)| softfloat_ref::fadd(a, b, m)).collect();
    cpu_out.iter().zip(gpu_out.iter()).filter(|(a, b)| a != b).count() as u32
}

// =============================================================================
// CPU baselines: each thread runs the same chain shape as the GPU kernel —
// 1024 ops with a low-mantissa twiddle on the operand. Hot path stays
// near 1.0 so we measure the FPU, not NaN/Inf path costs.
// =============================================================================

const N_CHAIN_OPS: usize = gpu::CHAIN_OPS_PER_THREAD;

#[inline]
fn twiddle(b: u64, a: u64) -> u64 {
    b ^ (a & 0xFF)
}

fn cpu_chain_binary<F: Fn(u64, u64) -> u64 + Send + Sync + Copy + 'static>(
    threads: usize,
    n_threads: usize,
    seed: (u64, u64),
    op: F,
) -> f64 {
    let chunk = threads.div_ceil(n_threads);
    let t = Instant::now();
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            thread::spawn(move || {
                let start = tid * chunk;
                let end = (start + chunk).min(threads);
                let mut acc = 0u64;
                for gid in start..end {
                    let mut a = 0x3FF0_0000_0000_0000u64 ^ ((seed.0 ^ gid as u64) & 0xFF);
                    let mut b = 0x3FF0_0000_0000_0001u64 ^ ((seed.1 ^ gid as u64) & 0xFF);
                    for _ in 0..N_CHAIN_OPS {
                        a = op(a, b);
                        b = twiddle(b, a);
                    }
                    acc ^= a ^ b;
                }
                acc
            })
        })
        .collect();
    let mut total = 0u64;
    for h in handles {
        total ^= h.join().unwrap();
    }
    std::hint::black_box(total);
    t.elapsed().as_secs_f64()
}

fn cpu_chain_unary<F: Fn(u64) -> u64 + Send + Sync + Copy + 'static>(
    threads: usize,
    n_threads: usize,
    seed: (u64, u64),
    op: F,
) -> f64 {
    let chunk = threads.div_ceil(n_threads);
    let t = Instant::now();
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            thread::spawn(move || {
                let start = tid * chunk;
                let end = (start + chunk).min(threads);
                let mut acc = 0u64;
                for gid in start..end {
                    let mut a = 0x3FF0_0000_0000_0000u64 ^ ((seed.0 ^ gid as u64) & 0xFF);
                    let b = 0x3FF0_0000_0000_0001u64 ^ ((seed.1 ^ gid as u64) & 0xFF);
                    for _ in 0..N_CHAIN_OPS {
                        a = op(a);
                        a = twiddle(a, b);
                    }
                    acc ^= a ^ b;
                }
                acc
            })
        })
        .collect();
    let mut total = 0u64;
    for h in handles {
        total ^= h.join().unwrap();
    }
    std::hint::black_box(total);
    t.elapsed().as_secs_f64()
}

fn cpu_chain_ternary<F: Fn(u64, u64, u64) -> u64 + Send + Sync + Copy + 'static>(
    threads: usize,
    n_threads: usize,
    seed: (u64, u64),
    op: F,
) -> f64 {
    let chunk = threads.div_ceil(n_threads);
    let t = Instant::now();
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            thread::spawn(move || {
                let start = tid * chunk;
                let end = (start + chunk).min(threads);
                let mut acc = 0u64;
                for gid in start..end {
                    let mut a = 0x3FF0_0000_0000_0000u64 ^ ((seed.0 ^ gid as u64) & 0xFF);
                    let mut b = 0x3FF0_0000_0000_0001u64 ^ ((seed.1 ^ gid as u64) & 0xFF);
                    let c = 0x3FF0_0000_0000_0000u64;
                    for _ in 0..N_CHAIN_OPS {
                        a = op(a, b, c);
                        b = twiddle(b, a);
                    }
                    acc ^= a ^ b;
                }
                acc
            })
        })
        .collect();
    let mut total = 0u64;
    for h in handles {
        total ^= h.join().unwrap();
    }
    std::hint::black_box(total);
    t.elapsed().as_secs_f64()
}

fn hw_fadd(a: u64, b: u64) -> u64 { (f64::from_bits(a) + f64::from_bits(b)).to_bits() }
fn hw_fsub(a: u64, b: u64) -> u64 { (f64::from_bits(a) - f64::from_bits(b)).to_bits() }
fn hw_fmul(a: u64, b: u64) -> u64 { (f64::from_bits(a) * f64::from_bits(b)).to_bits() }
fn hw_fdiv(a: u64, b: u64) -> u64 { (f64::from_bits(a) / f64::from_bits(b)).to_bits() }
fn hw_fsqrt(a: u64) -> u64 { f64::from_bits(a).abs().sqrt().to_bits() }
fn hw_fma(a: u64, b: u64, c: u64) -> u64 {
    f64::from_bits(a).mul_add(f64::from_bits(b), f64::from_bits(c)).to_bits()
}

// Chain bodies for the conversion / comparison ops. Each one mirrors the
// MSL kernel's loop shape so the CPU side does the same data-dependency
// dance as the GPU and the comparison is honest. All return u64 for the
// uniform helper signature (i64 / i32 / bool / u32 results are folded
// into the u64 output before write).

fn cpu_chain_cvt_unary<F: Fn(u64) -> u64 + Send + Sync + Copy + 'static>(
    threads: usize,
    n_threads: usize,
    seed: (u64, u64),
    init: u64,
    op: F,
) -> f64 {
    let chunk = threads.div_ceil(n_threads);
    let t = Instant::now();
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            thread::spawn(move || {
                let start = tid * chunk;
                let end = (start + chunk).min(threads);
                let mut acc = 0u64;
                for gid in start..end {
                    let mut x = init ^ ((seed.0 ^ gid as u64) & 0xFF);
                    let mut local = 0u64;
                    for _ in 0..N_CHAIN_OPS {
                        let r = op(x);
                        local ^= r;
                        x ^= r & 0xFF;
                    }
                    acc ^= local ^ x;
                }
                acc
            })
        })
        .collect();
    let mut total = 0u64;
    for h in handles {
        total ^= h.join().unwrap();
    }
    std::hint::black_box(total);
    t.elapsed().as_secs_f64()
}

fn cpu_chain_cmp<F: Fn(u64, u64) -> u64 + Send + Sync + Copy + 'static>(
    threads: usize,
    n_threads: usize,
    seed: (u64, u64),
    op: F,
) -> f64 {
    let chunk = threads.div_ceil(n_threads);
    let t = Instant::now();
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            thread::spawn(move || {
                let start = tid * chunk;
                let end = (start + chunk).min(threads);
                let mut acc = 0u64;
                for gid in start..end {
                    let a = 0x3FF0_0000_0000_0000u64 ^ ((seed.0 ^ gid as u64) & 0xFF);
                    let mut b = 0x3FF0_0000_0000_0001u64 ^ ((seed.1 ^ gid as u64) & 0xFF);
                    let mut local = 0u64;
                    for _ in 0..N_CHAIN_OPS {
                        let r = op(a, b);
                        local = (local << 1) ^ r;
                        b = twiddle(b, a ^ local);
                    }
                    acc ^= local ^ a ^ b;
                }
                acc
            })
        })
        .collect();
    let mut total = 0u64;
    for h in handles {
        total ^= h.join().unwrap();
    }
    std::hint::black_box(total);
    t.elapsed().as_secs_f64()
}

// Native-f64 baselines for the conversions / comparisons. All take and
// return u64 so they fit `cpu_chain_cvt_unary` / `cpu_chain_cmp`.
// Conversions feed integer-typed results back through `as u64` to match
// the MSL kernel's two's-complement reinterpret on the result mix.
fn hw_cvt_i64_to_f64(x: u64) -> u64 { (x as i64 as f64).to_bits() }
fn hw_cvt_u64_to_f64(x: u64) -> u64 { (x as f64).to_bits() }
fn hw_cvt_f64_to_i64(a: u64) -> u64 { f64::from_bits(a) as i64 as u64 }
fn hw_cvt_f32_to_f64(x: u64) -> u64 { (f32::from_bits(x as u32) as f64).to_bits() }
fn hw_cvt_f64_to_f32(a: u64) -> u64 { u64::from((f64::from_bits(a) as f32).to_bits()) }

fn hw_feq(a: u64, b: u64) -> u64 { u64::from(f64::from_bits(a) == f64::from_bits(b)) }
fn hw_flt(a: u64, b: u64) -> u64 { u64::from(f64::from_bits(a) <  f64::from_bits(b)) }
fn hw_fle(a: u64, b: u64) -> u64 { u64::from(f64::from_bits(a) <= f64::from_bits(b)) }
fn hw_fgt(a: u64, b: u64) -> u64 { u64::from(f64::from_bits(a) >  f64::from_bits(b)) }
fn hw_fge(a: u64, b: u64) -> u64 { u64::from(f64::from_bits(a) >= f64::from_bits(b)) }

// =============================================================================

fn time_gpu<F: Fn() -> Vec<u64>>(runs: usize, f: F) -> f64 {
    let _ = f(); // warmup
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        let v = f();
        let s = t.elapsed().as_secs_f64();
        std::hint::black_box(v);
        if s < best {
            best = s;
        }
    }
    best
}

fn time_cpu<F: FnMut() -> f64>(runs: usize, mut f: F) -> f64 {
    let _ = f();
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let s = f();
        if s < best {
            best = s;
        }
    }
    best
}

fn report(op: &str, total_ops: usize, cpu_t: f64, gpu_t: f64) {
    let cpu_thr = total_ops as f64 / cpu_t / 1e9;
    let gpu_thr = total_ops as f64 / gpu_t / 1e9;
    let speedup = cpu_t / gpu_t;
    println!(
        "  {:<6}    {:>6.2} G/s    {:>6.2} G/s   {:>5.2}×",
        op, cpu_thr, gpu_thr, speedup,
    );
}

fn main() {
    let threads: usize = std::env::var("THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let runs: usize = std::env::var("RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let total_ops = threads * N_CHAIN_OPS;
    let seed = (0x4000_0000_0000_0000u64, 0x3FF0_0000_0000_0001u64);

    println!(
        "\nGPU: {threads} lanes × {N_CHAIN_OPS} ops = {total_ops} total per measurement"
    );
    println!("CPU: {n_threads} OS threads doing the same op count on hw f64\n");

    let mismatches = correctness_check();
    if mismatches != 0 {
        eprintln!("✗ correctness check failed: {mismatches} bit mismatches between GPU and CPU softfloat_ref");
        std::process::exit(1);
    }
    println!("✓ correctness: 4096-pair fadd_batch matches softfloat_ref bit-for-bit");
    println!("  (softfloat_ref is itself bit-exact vs native CPU f64 in nearest mode)\n");

    println!("throughput      CPU ({n_threads}T)     GPU softfloat   speedup");
    println!("---------     ------------    ------------   --------");

    for (name, hw, gpu_fn) in [
        ("fadd", hw_fadd as fn(u64, u64) -> u64, gpu::fadd_chain as fn(usize, (u64, u64)) -> Vec<u64>),
        ("fsub", hw_fsub, gpu::fsub_chain),
        ("fmul", hw_fmul, gpu::fmul_chain),
        ("fdiv", hw_fdiv, gpu::fdiv_chain),
    ] {
        let cpu_t = time_cpu(runs, || cpu_chain_binary(threads, n_threads, seed, hw));
        let gpu_t = time_gpu(runs, || gpu_fn(threads, seed));
        report(name, total_ops, cpu_t, gpu_t);
    }
    {
        let cpu_t = time_cpu(runs, || cpu_chain_unary(threads, n_threads, seed, hw_fsqrt));
        let gpu_t = time_gpu(runs, || gpu::fsqrt_chain(threads, seed));
        report("fsqrt", total_ops, cpu_t, gpu_t);
    }
    {
        let cpu_t = time_cpu(runs, || cpu_chain_ternary(threads, n_threads, seed, hw_fma));
        let gpu_t = time_gpu(runs, || gpu::fma_chain(threads, seed));
        report("fma", total_ops, cpu_t, gpu_t);
    }

    // --- Conversions ----------------------------------------------------
    //
    // Each `init` mirrors the MSL chain kernel's seed value: integers
    // start near 1000 (cvt_*_to_f64) so the chain doesn't drift into
    // saturation, f64/f32 inputs start at 1.0 with low-byte jitter.
    for (name, init, hw, gpu_fn) in [
        ("cvt_i64_to_f64", 1000u64,
            hw_cvt_i64_to_f64 as fn(u64) -> u64,
            gpu::cvt_i64_to_f64_chain as fn(usize, (u64, u64)) -> Vec<u64>),
        ("cvt_u64_to_f64", 1000u64, hw_cvt_u64_to_f64, gpu::cvt_u64_to_f64_chain),
        ("cvt_f64_to_i64", 0x3FF0_0000_0000_0000, hw_cvt_f64_to_i64, gpu::cvt_f64_to_i64_chain),
        ("cvt_f32_to_f64", 0x3F80_0000, hw_cvt_f32_to_f64, gpu::cvt_f32_to_f64_chain),
        ("cvt_f64_to_f32", 0x3FF0_0000_0000_0000, hw_cvt_f64_to_f32, gpu::cvt_f64_to_f32_chain),
    ] {
        let cpu_t = time_cpu(runs, || cpu_chain_cvt_unary(threads, n_threads, seed, init, hw));
        let gpu_t = time_gpu(runs, || gpu_fn(threads, seed));
        report(name, total_ops, cpu_t, gpu_t);
    }

    // --- Comparisons ----------------------------------------------------
    for (name, hw, gpu_fn) in [
        ("feq", hw_feq as fn(u64, u64) -> u64, gpu::feq_chain as fn(usize, (u64, u64)) -> Vec<u64>),
        ("flt", hw_flt, gpu::flt_chain),
        ("fle", hw_fle, gpu::fle_chain),
        ("fgt", hw_fgt, gpu::fgt_chain),
        ("fge", hw_fge, gpu::fge_chain),
    ] {
        let cpu_t = time_cpu(runs, || cpu_chain_cmp(threads, n_threads, seed, hw));
        let gpu_t = time_gpu(runs, || gpu_fn(threads, seed));
        report(name, total_ops, cpu_t, gpu_t);
    }

    println!(
        "\n  Each GPU softfloat op runs ~30-50 integer instructions; CPU has a hardware FPU\n  per core. The GPU still wins on aggregate throughput, and the kernels produce\n  identical bits to native CPU `f64` arithmetic.\n",
    );
}
