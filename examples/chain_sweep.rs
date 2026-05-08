//! Sweep `__SOFTFLOAT64_CHAIN_OPS` across all six softfloat ops to find
//! the per-op sweet spot. Different ops have different per-call cost
//! (fmul/fdiv/fsqrt/fma each emit 30-100+ instructions vs fadd's ~30)
//! and different register pressure profiles, so the saturation knee can
//! move.
//!
//! Run:
//!
//! ```sh
//! cargo run --release --features gpu --example chain_sweep
//! ```

#![cfg(target_os = "macos")]

use std::time::Instant;

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use metal_softfloat::METAL_SOURCE;

const SWEEP_POINTS: &[u32] = &[16, 64, 256, 512, 1024, 2048, 4096, 8192];
const THREADS: usize = 200_000;
const RUNS: usize = 3;

/// Build a one-off chain kernel for `op_name` at compile-time chain
/// length `n_ops`. The kernel body is inlined into the dist source so
/// the for-loop iteration count is a true compile-time constant.
fn shader_with_n(op_name: &str, n_ops: u32) -> String {
    // Body shape per op kind: binary (fadd/fsub/fmul/fdiv) takes (a, b),
    // unary (fsqrt) takes (a), ternary (fma) takes (a, b, c).
    let body = match op_name {
        "fadd" => "a = __softfloat64_fadd(a, b, 0u); b ^= a & 0xFFUL;",
        "fsub" => "a = __softfloat64_fsub(a, b, 0u); b ^= a & 0xFFUL;",
        "fmul" => "a = __softfloat64_fmul(a, b, 0u); b ^= a & 0xFFUL;",
        "fdiv" => "a = __softfloat64_fdiv(a, b, 0u); b ^= a & 0xFFUL;",
        "fsqrt" => "a = __softfloat64_fsqrt(a, 0u); a ^= b ^ ((ulong)i << 32);",
        "fma" => "a = __softfloat64_fma(a, b, c, 0u); b ^= a & 0xFFUL;",
        _ => unreachable!(),
    };
    let extra_init = if op_name == "fma" {
        "ulong c = 0x3FF0000000000000UL;"
    } else {
        ""
    };
    format!(
        "{base}\n\
        kernel void chain_n(\n\
            constant ulong2& seed [[buffer(0)]],\n\
            device ulong*    out  [[buffer(1)]],\n\
            uint gid [[thread_position_in_grid]])\n\
        {{\n\
            ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);\n\
            ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);\n\
            {extra_init}\n\
            #pragma unroll 2\n\
            for (uint i = 0; i < {n_ops}u; ++i) {{\n\
                {body}\n\
            }}\n\
            out[gid] = a ^ b;\n\
        }}\n",
        base = METAL_SOURCE,
    )
}

fn time_dispatch(device: &Device, queue: &metal::CommandQueue, op: &str, n_ops: u32) -> f64 {
    let src = shader_with_n(op, n_ops);
    let library = device
        .new_library_with_source(&src, &CompileOptions::new())
        .expect("compile shader");
    let function = library.get_function("chain_n", None).unwrap();
    let pso = device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap();

    let seed: [u64; 2] = [0x4000_0000_0000_0000, 0x3FF0_0000_0000_0001];
    let out_buf =
        device.new_buffer((THREADS * 8) as u64, MTLResourceOptions::StorageModeShared);
    let dispatch = || {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_bytes(0, 16, seed.as_ptr().cast());
        enc.set_buffer(1, Some(&out_buf), 0);
        let tg = pso.thread_execution_width();
        enc.dispatch_threads(
            MTLSize::new(THREADS as u64, 1, 1),
            MTLSize::new(tg.min(THREADS as u64), 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    };
    dispatch(); // warmup
    let mut best = f64::INFINITY;
    for _ in 0..RUNS {
        let t = Instant::now();
        dispatch();
        let s = t.elapsed().as_secs_f64();
        if s < best {
            best = s;
        }
    }
    best
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();

    println!(
        "\nSweep: chain ops over {:?}, all six ops at {} GPU lanes",
        SWEEP_POINTS, THREADS
    );
    println!("Per-cell value is aggregate G ops/sec (higher = better).\n");

    print!("  N_ops ");
    for op in &["fadd", "fsub", "fmul", "fdiv", "fsqrt", "fma"] {
        print!(" {:>7}", op);
    }
    println!();
    print!("--------");
    for _ in 0..6 {
        print!(" -------");
    }
    println!();

    // Per-op asymptotic best, used to print percent-of-best in a second pass.
    let mut measured: Vec<(u32, [f64; 6])> = Vec::with_capacity(SWEEP_POINTS.len());
    for &n in SWEEP_POINTS {
        let mut row = [0.0; 6];
        for (i, op) in ["fadd", "fsub", "fmul", "fdiv", "fsqrt", "fma"]
            .iter()
            .enumerate()
        {
            let t = time_dispatch(&device, &queue, op, n);
            let total = THREADS * n as usize;
            row[i] = total as f64 / t / 1e9;
        }
        print!("  {:>5} ", n);
        for v in &row {
            print!(" {:>7.2}", v);
        }
        println!();
        measured.push((n, row));
    }

    // Per-op best across the sweep.
    let bests: [f64; 6] = {
        let mut b = [0.0; 6];
        for (_, row) in &measured {
            for i in 0..6 {
                if row[i] > b[i] {
                    b[i] = row[i];
                }
            }
        }
        b
    };

    println!("\n  N_ops  | percent of per-op best");
    print!("--------+");
    for _ in 0..6 {
        print!(" -------");
    }
    println!();
    print!("  best   ");
    for v in &bests {
        print!(" {:>6.2}G", v);
    }
    println!();
    for (n, row) in &measured {
        print!("  {:>5}  ", n);
        for i in 0..6 {
            print!(" {:>6.1}%", 100.0 * row[i] / bests[i]);
        }
        println!();
    }

    println!(
        "\n  Sweet spot is the smallest N where every op is at >=99% of\n  its asymptotic best. Larger N just makes each `*_chain` call\n  slower without buying more throughput.\n",
    );
}
