//! Berkeley TestFloat conformance test for `softfloat_ref`.
//!
//! Opt-in via `--features testfloat`. Requires Berkeley TestFloat-3e
//! (http://www.jhauser.us/arithmetic/TestFloat.html) — the test locates
//! `testfloat_gen` first via the `TESTFLOAT_GEN` environment variable,
//! then on `PATH`. If absent, the test fails with install instructions.
//!
//! For each binary64 op × rounding mode the test spawns `testfloat_gen`,
//! parses its `arg ... result flags` lines, runs the corresponding
//! `softfloat_ref` op, and asserts bit equality. NaN payload differences
//! are tolerated: this crate canonicalizes every NaN result to qNaN
//! `0x7FF8_0000_0000_0000`, while TestFloat may emit other valid NaN
//! payloads. Status flags are not checked — this crate does not expose
//! an fenv-style flag set.
//!
//! `TESTFLOAT_LEVEL=2` opts into thorough mode (slow). Default is level 1.
//!
//! Subnormal-flush (`ftz`) is incompatible with TestFloat's IEEE-754
//! §7.4 vectors, so this test is gated off when `ftz` is enabled.

#![cfg(all(feature = "testfloat", not(feature = "ftz")))]

use metal_softfloat::{softfloat_ref, RoundingMode};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

fn testfloat_gen_path() -> String {
    if let Ok(p) = std::env::var("TESTFLOAT_GEN") {
        return p;
    }
    if let Ok(out) = Command::new("which").arg("testfloat_gen").output() {
        if out.status.success() {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !p.is_empty() {
                return p;
            }
        }
    }
    panic!(
        "testfloat_gen not found. Build Berkeley TestFloat-3e from \
         http://www.jhauser.us/arithmetic/TestFloat-3e.zip (depends on \
         SoftFloat-3e), then put `testfloat_gen` on PATH or set \
         TESTFLOAT_GEN=/abs/path/to/testfloat_gen."
    );
}

fn level_arg() -> String {
    std::env::var("TESTFLOAT_LEVEL").unwrap_or_else(|_| "1".to_string())
}

fn rmode_arg(mode: RoundingMode) -> &'static str {
    match mode {
        RoundingMode::Nearest => "-rnear_even",
        RoundingMode::Down => "-rmin",
        RoundingMode::Up => "-rmax",
        RoundingMode::Zero => "-rminMag",
    }
}

fn parse_u64_hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16)
        .unwrap_or_else(|_| panic!("invalid hex u64 from testfloat_gen: {s:?}"))
}

fn is_nan(bits: u64) -> bool {
    matches!(softfloat_ref::classify(bits), softfloat_ref::IeeeClass::NaN)
}

/// Bit-equal, with the documented exception that any-NaN equals any-NaN.
fn results_match(got: u64, expected: u64) -> bool {
    got == expected || (is_nan(got) && is_nan(expected))
}

const MAX_REPORTED: usize = 16;

fn run_op<F>(op: &str, mode: RoundingMode, arity: usize, run: F)
where
    F: Fn(&[u64], RoundingMode) -> u64,
{
    let bin = testfloat_gen_path();
    let mut child = Command::new(&bin)
        .arg(rmode_arg(mode))
        .arg("-level")
        .arg(level_arg())
        .arg(op)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {bin}: {e}"));

    let stdout = child.stdout.take().expect("stdout");
    let reader = BufReader::new(stdout);

    let mut total: u64 = 0;
    let mut mismatches: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => panic!("read {op} stdout: {e}"),
        };
        let toks: Vec<&str> = line.split_whitespace().collect();
        // Each TestFloat line is: <arity hex args> <hex result> <hex flags>.
        if toks.len() < arity + 2 {
            continue;
        }
        let args: Vec<u64> = toks[..arity].iter().copied().map(parse_u64_hex).collect();
        let expected = parse_u64_hex(toks[arity]);
        let got = run(&args, mode);
        total += 1;
        if !results_match(got, expected) && mismatches.len() < MAX_REPORTED {
            let arg_str = args
                .iter()
                .map(|b| format!("{b:016x}"))
                .collect::<Vec<_>>()
                .join(", ");
            mismatches.push(format!(
                "args=[{arg_str}] expected={expected:016x} got={got:016x}"
            ));
        }
    }

    let status = child.wait().expect("wait testfloat_gen");
    assert!(status.success(), "testfloat_gen {op} {mode:?} exited non-zero");
    assert!(total > 0, "testfloat_gen {op} {mode:?} produced no vectors");

    assert!(
        mismatches.is_empty(),
        "{} mismatches against TestFloat (op={op}, mode={mode:?}, vectors={total}). \
         first {}:\n  {}",
        mismatches.len(),
        mismatches.len(),
        mismatches.join("\n  ")
    );

    eprintln!("[testfloat] {op} {mode:?}: {total} vectors OK");
}

// One #[test] per (op, mode). Cargo's default test runner runs `#[test]`
// functions in parallel (up to `num_cpus`), so 24 leaf tests fan out
// across cores without any rayon dependency. The slow op (fma) ends up
// running its four rounding modes concurrently instead of serially —
// roughly 4× wall-clock improvement on its own.
macro_rules! conformance {
    ($name:ident, $op:literal, $mode:expr, $arity:expr, $run:expr) => {
        #[test]
        fn $name() {
            run_op($op, $mode, $arity, $run);
        }
    };
}

conformance!(f64_add_nearest, "f64_add", RoundingMode::Nearest, 2, |a, m| softfloat_ref::fadd(a[0], a[1], m));
conformance!(f64_add_down,    "f64_add", RoundingMode::Down,    2, |a, m| softfloat_ref::fadd(a[0], a[1], m));
conformance!(f64_add_up,      "f64_add", RoundingMode::Up,      2, |a, m| softfloat_ref::fadd(a[0], a[1], m));
conformance!(f64_add_zero,    "f64_add", RoundingMode::Zero,    2, |a, m| softfloat_ref::fadd(a[0], a[1], m));

conformance!(f64_sub_nearest, "f64_sub", RoundingMode::Nearest, 2, |a, m| softfloat_ref::fsub(a[0], a[1], m));
conformance!(f64_sub_down,    "f64_sub", RoundingMode::Down,    2, |a, m| softfloat_ref::fsub(a[0], a[1], m));
conformance!(f64_sub_up,      "f64_sub", RoundingMode::Up,      2, |a, m| softfloat_ref::fsub(a[0], a[1], m));
conformance!(f64_sub_zero,    "f64_sub", RoundingMode::Zero,    2, |a, m| softfloat_ref::fsub(a[0], a[1], m));

conformance!(f64_mul_nearest, "f64_mul", RoundingMode::Nearest, 2, |a, m| softfloat_ref::fmul(a[0], a[1], m));
conformance!(f64_mul_down,    "f64_mul", RoundingMode::Down,    2, |a, m| softfloat_ref::fmul(a[0], a[1], m));
conformance!(f64_mul_up,      "f64_mul", RoundingMode::Up,      2, |a, m| softfloat_ref::fmul(a[0], a[1], m));
conformance!(f64_mul_zero,    "f64_mul", RoundingMode::Zero,    2, |a, m| softfloat_ref::fmul(a[0], a[1], m));

conformance!(f64_div_nearest, "f64_div", RoundingMode::Nearest, 2, |a, m| softfloat_ref::fdiv(a[0], a[1], m));
conformance!(f64_div_down,    "f64_div", RoundingMode::Down,    2, |a, m| softfloat_ref::fdiv(a[0], a[1], m));
conformance!(f64_div_up,      "f64_div", RoundingMode::Up,      2, |a, m| softfloat_ref::fdiv(a[0], a[1], m));
conformance!(f64_div_zero,    "f64_div", RoundingMode::Zero,    2, |a, m| softfloat_ref::fdiv(a[0], a[1], m));

conformance!(f64_sqrt_nearest, "f64_sqrt", RoundingMode::Nearest, 1, |a, m| softfloat_ref::fsqrt(a[0], m));
conformance!(f64_sqrt_down,    "f64_sqrt", RoundingMode::Down,    1, |a, m| softfloat_ref::fsqrt(a[0], m));
conformance!(f64_sqrt_up,      "f64_sqrt", RoundingMode::Up,      1, |a, m| softfloat_ref::fsqrt(a[0], m));
conformance!(f64_sqrt_zero,    "f64_sqrt", RoundingMode::Zero,    1, |a, m| softfloat_ref::fsqrt(a[0], m));

conformance!(f64_muladd_nearest, "f64_mulAdd", RoundingMode::Nearest, 3, |a, m| softfloat_ref::fma(a[0], a[1], a[2], m));
conformance!(f64_muladd_down,    "f64_mulAdd", RoundingMode::Down,    3, |a, m| softfloat_ref::fma(a[0], a[1], a[2], m));
conformance!(f64_muladd_up,      "f64_mulAdd", RoundingMode::Up,      3, |a, m| softfloat_ref::fma(a[0], a[1], a[2], m));
conformance!(f64_muladd_zero,    "f64_mulAdd", RoundingMode::Zero,    3, |a, m| softfloat_ref::fma(a[0], a[1], a[2], m));

// --- Conversions ---------------------------------------------------------
//
// Result encodings: f64→i64 reinterprets the i64 bit pattern as u64;
// f64→f32 zero-extends the u32 result; comparisons emit 0/1 as u64.
// `f32_to_f64` is exact and ignores rmode but TestFloat still requires
// one of `-rnear_even` etc. — pass Nearest as the canonical filler.

conformance!(i64_to_f64_nearest, "i64_to_f64", RoundingMode::Nearest, 1, |a, m| softfloat_ref::cvt_i64_to_f64(a[0] as i64, m));
conformance!(i64_to_f64_down,    "i64_to_f64", RoundingMode::Down,    1, |a, m| softfloat_ref::cvt_i64_to_f64(a[0] as i64, m));
conformance!(i64_to_f64_up,      "i64_to_f64", RoundingMode::Up,      1, |a, m| softfloat_ref::cvt_i64_to_f64(a[0] as i64, m));
conformance!(i64_to_f64_zero,    "i64_to_f64", RoundingMode::Zero,    1, |a, m| softfloat_ref::cvt_i64_to_f64(a[0] as i64, m));

conformance!(ui64_to_f64_nearest, "ui64_to_f64", RoundingMode::Nearest, 1, |a, m| softfloat_ref::cvt_u64_to_f64(a[0], m));
conformance!(ui64_to_f64_down,    "ui64_to_f64", RoundingMode::Down,    1, |a, m| softfloat_ref::cvt_u64_to_f64(a[0], m));
conformance!(ui64_to_f64_up,      "ui64_to_f64", RoundingMode::Up,      1, |a, m| softfloat_ref::cvt_u64_to_f64(a[0], m));
conformance!(ui64_to_f64_zero,    "ui64_to_f64", RoundingMode::Zero,    1, |a, m| softfloat_ref::cvt_u64_to_f64(a[0], m));

conformance!(f64_to_i64_nearest, "f64_to_i64", RoundingMode::Nearest, 1, |a, m| softfloat_ref::cvt_f64_to_i64(a[0], m) as u64);
conformance!(f64_to_i64_down,    "f64_to_i64", RoundingMode::Down,    1, |a, m| softfloat_ref::cvt_f64_to_i64(a[0], m) as u64);
conformance!(f64_to_i64_up,      "f64_to_i64", RoundingMode::Up,      1, |a, m| softfloat_ref::cvt_f64_to_i64(a[0], m) as u64);
conformance!(f64_to_i64_zero,    "f64_to_i64", RoundingMode::Zero,    1, |a, m| softfloat_ref::cvt_f64_to_i64(a[0], m) as u64);

conformance!(f32_to_f64_exact, "f32_to_f64", RoundingMode::Nearest, 1, |a, _m| softfloat_ref::cvt_f32_to_f64(a[0] as u32));

conformance!(f64_to_f32_nearest, "f64_to_f32", RoundingMode::Nearest, 1, |a, m| u64::from(softfloat_ref::cvt_f64_to_f32(a[0], m)));
conformance!(f64_to_f32_down,    "f64_to_f32", RoundingMode::Down,    1, |a, m| u64::from(softfloat_ref::cvt_f64_to_f32(a[0], m)));
conformance!(f64_to_f32_up,      "f64_to_f32", RoundingMode::Up,      1, |a, m| u64::from(softfloat_ref::cvt_f64_to_f32(a[0], m)));
conformance!(f64_to_f32_zero,    "f64_to_f32", RoundingMode::Zero,    1, |a, m| u64::from(softfloat_ref::cvt_f64_to_f32(a[0], m)));

// --- Comparisons ---------------------------------------------------------
//
// Mode-independent. Run each at Nearest (TestFloat ignores rmode for
// these). We map our IEEE-quiet implementation to TestFloat's `_quiet`
// variants for le/lt; `f64_eq` is already quiet semantics in TestFloat.

conformance!(f64_eq_quiet, "f64_eq",       RoundingMode::Nearest, 2, |a, _m| u64::from(softfloat_ref::feq(a[0], a[1])));
conformance!(f64_le_quiet, "f64_le_quiet", RoundingMode::Nearest, 2, |a, _m| u64::from(softfloat_ref::fle(a[0], a[1])));
conformance!(f64_lt_quiet, "f64_lt_quiet", RoundingMode::Nearest, 2, |a, _m| u64::from(softfloat_ref::flt(a[0], a[1])));
