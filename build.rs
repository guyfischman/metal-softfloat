// Build script for metal-softfloat-core.
//
// Only does work when the `testfloat-gpu` feature is enabled. In that
// case it compiles the vendored Berkeley SoftFloat-3e + TestFloat-3e
// case-generation sources into a static library that the
// `tests/testfloat_gpu_conformance.rs` integration test links against.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_TESTFLOAT_GPU").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sf_root = manifest_dir.join("vendor/SoftFloat-3e");
    let sf_source = sf_root.join("source");
    let sf_8086 = sf_source.join("8086-SSE");
    let sf_include = sf_source.join("include");

    let tf_root = manifest_dir.join("vendor/TestFloat-3e");
    let tf_source = tf_root.join("source");

    // Mirror SoftFloat-3e/build/Linux-x86_64-GCC/Makefile flags. The
    // `8086-SSE` specialization sets canonical-NaN bit patterns that
    // happen to match what this crate uses (qNaN bit 51 set, payload 0).
    let mut sf = cc::Build::new();
    sf.include(&sf_root) // platform.h
        .include(&sf_8086) // specialize.h
        .include(&sf_include) // softfloat.h, internals.h, primitives.h, ...
        .define("SOFTFLOAT_FAST_INT64", None)
        .define("SOFTFLOAT_ROUND_ODD", None)
        .define("INLINE_LEVEL", "5")
        .define("SOFTFLOAT_FAST_DIV32TO16", None)
        .define("SOFTFLOAT_FAST_DIV64TO32", None)
        .opt_level(2)
        .warnings(false);

    // SoftFloat top-level sources we need.
    for f in [
        // f64 arithmetic
        "f64_add.c",
        "f64_sub.c",
        "f64_mul.c",
        "f64_div.c",
        "f64_sqrt.c",
        "f64_mulAdd.c",
        "s_addMagsF64.c",
        "s_subMagsF64.c",
        "s_mulAddF64.c",
        "s_normSubnormalF64Sig.c",
        "s_normRoundPackToF64.c",
        "s_roundPackToF64.c",
        // f64 conversions and comparisons
        "i64_to_f64.c",
        "ui64_to_f64.c",
        "f64_to_i64.c",
        "f32_to_f64.c",
        "f64_to_f32.c",
        "f64_eq.c",
        "f64_lt_quiet.c",
        "f64_le_quiet.c",
        "s_roundToI64.c",
        // f32 helpers (for f32↔f64 conversions)
        "s_normSubnormalF32Sig.c",
        "s_roundPackToF32.c",
        "s_normRoundPackToF32.c",
        // shift / count primitives
        "s_shiftRightJam32.c",
        "s_shiftRightJam64.c",
        "s_shiftRightJam64Extra.c",
        "s_shiftRightJam128.c",
        "s_shiftRightJam128Extra.c",
        "s_shortShiftRightJam64.c",
        "s_shortShiftRightJam64Extra.c",
        "s_shortShiftRightJam128.c",
        "s_shortShiftRightJam128Extra.c",
        "s_countLeadingZeros8.c",
        "s_countLeadingZeros16.c",
        "s_countLeadingZeros32.c",
        "s_countLeadingZeros64.c",
        "s_approxRecip32_1.c",
        "s_approxRecipSqrt32_1.c",
        "s_approxRecip_1Ks.c",
        "s_approxRecipSqrt_1Ks.c",
        "s_mul64To128.c",
        "s_mul64ByShifted32To128.c",
        "s_add128.c",
        "s_sub128.c",
        "s_eq128.c",
        "s_le128.c",
        "s_lt128.c",
        "s_shortShiftLeft128.c",
        "s_shortShiftRight128.c",
        "softfloat_state.c",
    ] {
        sf.file(sf_source.join(f));
    }
    // 8086-SSE specialization (NaN handling + raiseFlags stub).
    for f in [
        "s_propagateNaNF64UI.c",
        "s_f64UIToCommonNaN.c",
        "s_commonNaNToF64UI.c",
        "s_propagateNaNF32UI.c",
        "s_f32UIToCommonNaN.c",
        "s_commonNaNToF32UI.c",
        "softfloat_raiseFlags.c",
    ] {
        sf.file(sf_8086.join(f));
    }

    println!("cargo:rerun-if-changed=vendor/SoftFloat-3e");
    sf.compile("softfloat3e");

    // TestFloat-3e case generators. Need its own platform.h and to see
    // SoftFloat's headers for `float64_t` typedef.
    let mut tf = cc::Build::new();
    tf.include(&tf_root) // platform.h
        .include(&tf_source) // genCases.h, fail.h, random.h
        .include(&sf_include) // softfloat.h, softfloat_types.h
        .define("FLOAT64", None)
        .opt_level(2)
        .warnings(false);
    for f in [
        "genCases_common.c",
        "genCases_f64.c",
        "genCases_f32.c",
        "genCases_i64.c",
        "genCases_ui64.c",
        "random.c",
        "fail.c",
    ] {
        tf.file(tf_source.join(f));
    }
    println!("cargo:rerun-if-changed=vendor/TestFloat-3e");
    tf.compile("testfloat3e_gen");
}
