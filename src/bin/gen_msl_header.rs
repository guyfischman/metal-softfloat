//! Generator for `dist/softfloat64.metal`.
//!
//! Reads `shaders/softfloat.metal` (the canonical MSL source — what the
//! Rust crate's GPU dispatchers `include_str!` and what every test runs
//! against), strips the in-tree test/benchmark scaffolding (everything
//! from the `// --- Test / benchmark kernels ---` banner onward), and
//! prepends the redistribution header (license + brief usage notes).
//! The result is the strict public API surface only.
//!
//! Run with:
//!
//! ```sh
//! cargo run --bin gen-msl-header
//! ```
//!
//! CI invokes this with `-- --check` to catch drift between
//! `shaders/softfloat.metal` and `dist/softfloat64.metal`. Do not
//! hand-edit `dist/softfloat64.metal`.

use std::fs;
use std::path::PathBuf;

const HEADER: &str = "\
// dist/softfloat64.metal — redistributable IEEE-754 binary64 softfloat
// for Apple Metal. Generated from metal-softfloat's
// shaders/softfloat.metal; do not hand-edit. Re-run the generator with
// `cargo run -p metal-softfloat --bin gen-msl-header`.
//
// SPDX-License-Identifier: BSD-3-Clause AND MIT
//
// Berkeley SoftFloat lookup tables (APPROX_RECIP_SQRT_K0S/K1S,
// APPROX_RECIP_K0S/K1S) and the recip / recipSqrt approximation
// functions are derived from John R. Hauser's Berkeley SoftFloat-3e:
//
//   Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017
//   The Regents of the University of California.
//   All Rights Reserved.
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions
//   are met:
//
//    1. Redistributions of source code must retain the above copyright
//       notice, this list of conditions, and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions, and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//    3. Neither the name of the University nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
//   THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS \"AS IS\",
//   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
//   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//   PARTICULAR PURPOSE, ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
//   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//   OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//   SUCH DAMAGE.
//
// All other contributions (the integer-arithmetic dispatchers, IEEE-754
// special-case heads, gradual-underflow code, FMA, MSL kernel layout,
// and tests) are MIT-licensed:
//
//   Copyright (c) 2026 metal-softfloat contributors.
//   See LICENSE for the full text.
//
// Public API:
//
//   ulong __softfloat64_fadd(ulong a, ulong b, uint mode);
//   ulong __softfloat64_fsub(ulong a, ulong b, uint mode);
//   ulong __softfloat64_fmul(ulong a, ulong b, uint mode);
//   ulong __softfloat64_fdiv(ulong a, ulong b, uint mode);
//   ulong __softfloat64_fsqrt(ulong a, uint mode);
//   ulong __softfloat64_fma(ulong a, ulong b, ulong c, uint mode);
//
//   ulong __softfloat64_cvt_i64_to_f64(long  x, uint mode);
//   ulong __softfloat64_cvt_u64_to_f64(ulong x, uint mode);
//   long  __softfloat64_cvt_f64_to_i64(ulong a, uint mode);
//   ulong __softfloat64_cvt_f32_to_f64(uint  a);                // exact, no mode
//   uint  __softfloat64_cvt_f64_to_f32(ulong a, uint mode);
//
//   bool  __softfloat64_feq(ulong a, ulong b);
//   bool  __softfloat64_flt(ulong a, ulong b);
//   bool  __softfloat64_fle(ulong a, ulong b);
//   bool  __softfloat64_fgt(ulong a, ulong b);
//   bool  __softfloat64_fge(ulong a, ulong b);
//
// `mode` is the IEEE-754 rounding mode:
//   0 = nearest-ties-to-even, 1 = toward -inf, 2 = toward +inf, 3 = toward zero.
//
// f64 / u64 / i64 values cross the API as `ulong` / `long` bit patterns;
// f32 values cross as `uint`. Use `as_type<>(...)` on the caller side to
// bit-cast between native floats and these integer payloads.
//
// Compile-time switches:
//   -DSOFTFLOAT_FTZ  flush subnormal inputs/outputs to zero.
//
// IEEE-754 conformance: see metal-softfloat/docs/ieee754_conformance.md.
// Limitation: __softfloat64_fdiv / __softfloat64_fsqrt currently flush
// subnormal *outputs* to zero. Subnormal inputs are handled correctly.
// Other ops (fadd, fsub, fmul, fma) produce gradual-underflow output.

";

// Anchor that marks the start of the in-tree test/benchmark scaffolding
// in `shaders/softfloat.metal`. Everything from this banner onward is
// stripped from the redistributable dist file.
const TEST_SCAFFOLDING_ANCHOR: &str = "// --- Test / benchmark kernels";

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = manifest.join("shaders/softfloat.metal");
    let dst = manifest.join("dist/softfloat64.metal");

    let body = fs::read_to_string(&src).expect("read shaders/softfloat.metal");
    let public = body
        .split_once(TEST_SCAFFOLDING_ANCHOR)
        .map_or(body.as_str(), |(before, _)| before)
        .trim_end();

    let mut out = String::with_capacity(HEADER.len() + public.len() + 1);
    out.push_str(HEADER);
    out.push_str(public);
    out.push('\n');

    let check_only = std::env::args().any(|a| a == "--check");
    if check_only {
        let existing = fs::read_to_string(&dst).unwrap_or_default();
        if existing != out {
            eprintln!(
                "{} is stale; re-run `cargo run -p metal-softfloat --bin gen-msl-header`",
                dst.display()
            );
            std::process::exit(1);
        }
        println!("{} matches generated content", dst.display());
        return;
    }
    fs::write(&dst, &out).expect("write dist/softfloat64.metal");
    println!("wrote {} ({} bytes)", dst.display(), out.len());
}
