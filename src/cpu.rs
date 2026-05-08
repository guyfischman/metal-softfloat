//! CPU reference implementation of the f64 ops.
//!
//! Uses the native `f64` type and (on aarch64) hardware FPCR control
//! for the rounding mode. The output of these functions *is* the
//! ground truth the GPU kernel has to match.
//!
//! Functions take and return `u64` bit patterns (`f64::to_bits` /
//! `from_bits`) to make byte-for-byte comparison with a Metal-produced
//! result trivial. No `f64` value ever crosses a public boundary.

use super::RoundingMode;

// --- Rounding-mode control (aarch64) --------------------------------------

#[cfg(target_arch = "aarch64")]
mod rmode {
    use super::RoundingMode;
    use core::arch::asm;

    const RMODE_SHIFT: u64 = 22;
    const RMODE_MASK: u64 = 0b11 << RMODE_SHIFT;

    /// Map the logical rounding mode to the ARM FPCR.RMode field. Note
    /// that bits 0 and 1 of our `RoundingMode` enum swap compared to
    /// ARM's encoding: 0→0b00 (nearest), 1→0b10 (down), 2→0b01 (up),
    /// 3→0b11 (zero). ARM's RMode: 00 RN, 01 RP (+∞), 10 RM (−∞), 11 RZ.
    const fn to_fpcr_bits(m: RoundingMode) -> u64 {
        match m {
            RoundingMode::Nearest => 0b00,
            RoundingMode::Down => 0b10,
            RoundingMode::Up => 0b01,
            RoundingMode::Zero => 0b11,
        }
    }

    // FPCR-write ordering is pinned by `compiler_fence(SeqCst)` on each
    // side of `Guard::new` and `Guard::drop` (below) — that's a full
    // barrier at the LLVM IR level, so the optimizer can't hoist FP
    // ops across the FPCR write or sink them past the restore. Inline
    // asm by itself is *not* an ordering barrier for surrounding ops,
    // which is the trap `std::hint::black_box` tries (insufficiently)
    // to address; `compiler_fence` is the correct primitive.

    fn read_fpcr() -> u64 {
        let v: u64;
        // SAFETY: `mrs ..., fpcr` reads a system register — no memory
        // effects, no side effects on other registers.
        unsafe { asm!("mrs {}, fpcr", out(reg) v, options(nomem, nostack, preserves_flags)) };
        v
    }

    fn write_fpcr(v: u64) {
        // SAFETY: `msr fpcr, ...` writes the FP control register on the
        // current thread; the new mode applies to subsequent FP ops
        // ordered against this asm by the surrounding compiler_fence.
        unsafe { asm!("msr fpcr, {}", in(reg) v, options(nomem, nostack, preserves_flags)) };
    }

    /// RAII guard that sets the FPCR rounding mode on construction and
    /// restores it on drop. Use it to wrap a block of FP ops so their
    /// results are deterministic under `mode`.
    pub struct Guard {
        saved: u64,
    }

    impl Guard {
        pub fn new(mode: RoundingMode) -> Self {
            use core::sync::atomic::{compiler_fence, Ordering};
            // Pre-fence: any FP op the caller emitted before constructing
            // the Guard must finish before the FPCR write.
            compiler_fence(Ordering::SeqCst);
            let saved = read_fpcr();
            let new = (saved & !RMODE_MASK) | (to_fpcr_bits(mode) << RMODE_SHIFT);
            write_fpcr(new);
            // Post-fence: any FP op after the Guard must observe the new
            // FPCR; no hoisting above the write.
            compiler_fence(Ordering::SeqCst);
            Self { saved }
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            use core::sync::atomic::{compiler_fence, Ordering};
            // Pre-fence: FP ops inside the guarded scope finish before
            // we restore the old FPCR.
            compiler_fence(Ordering::SeqCst);
            write_fpcr(self.saved);
            // Post-fence: subsequent FP ops can't be hoisted above the
            // restore.
            compiler_fence(Ordering::SeqCst);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod rmode {
    use super::RoundingMode;

    pub struct Guard;

    impl Guard {
        pub fn new(mode: RoundingMode) -> Self {
            // We only have FPCR access on aarch64. Silently running with
            // the wrong rounding mode would mask correctness bugs, so
            // panic if a directed mode is requested elsewhere. Callers
            // who actually need cross-platform directed rounding should
            // use `softfloat_ref` instead — it does the rounding in
            // software and runs identically on every target.
            assert!(
                matches!(mode, RoundingMode::Nearest),
                "cpu::* only supports RoundingMode::Nearest off aarch64; \
                 use softfloat_ref::* for portable directed rounding",
            );
            Self
        }
    }
}

// --- FP ops ----------------------------------------------------------------

/// `f64::from_bits(a) + f64::from_bits(b)` rounded as `mode` says,
/// returned as bits.
#[must_use]
pub fn fadd(a: u64, b: u64, mode: RoundingMode) -> u64 {
    let _g = rmode::Guard::new(mode);
    let x = std::hint::black_box(f64::from_bits(a));
    let y = std::hint::black_box(f64::from_bits(b));
    std::hint::black_box(x + y).to_bits()
}

#[must_use]
pub fn fsub(a: u64, b: u64, mode: RoundingMode) -> u64 {
    let _g = rmode::Guard::new(mode);
    let x = std::hint::black_box(f64::from_bits(a));
    let y = std::hint::black_box(f64::from_bits(b));
    std::hint::black_box(x - y).to_bits()
}

#[must_use]
pub fn fmul(a: u64, b: u64, mode: RoundingMode) -> u64 {
    let _g = rmode::Guard::new(mode);
    let x = std::hint::black_box(f64::from_bits(a));
    let y = std::hint::black_box(f64::from_bits(b));
    std::hint::black_box(x * y).to_bits()
}

#[must_use]
pub fn fdiv(a: u64, b: u64, mode: RoundingMode) -> u64 {
    let _g = rmode::Guard::new(mode);
    let x = std::hint::black_box(f64::from_bits(a));
    let y = std::hint::black_box(f64::from_bits(b));
    std::hint::black_box(x / y).to_bits()
}

#[must_use]
pub fn fsqrt(a: u64, mode: RoundingMode) -> u64 {
    let _g = rmode::Guard::new(mode);
    let x = std::hint::black_box(f64::from_bits(a));
    std::hint::black_box(x.sqrt()).to_bits()
}

/// IEEE-754 f64 bit pattern for an i32, rounding irrelevant because
/// every i32 is exactly representable. Matches `(x as f64).to_bits()`.
#[must_use]
pub fn cvt_i32_to_f64(x: i32) -> u64 {
    (f64::from(x)).to_bits()
}

/// IEEE-754 fused multiply-add `(a × b) + c` via native `f64::mul_add`.
/// On aarch64 this lowers to the hardware FMA instruction (single
/// rounding); ground truth for softfloat_ref::fma.
#[must_use]
pub fn fma(a: u64, b: u64, c: u64, mode: RoundingMode) -> u64 {
    let _g = rmode::Guard::new(mode);
    let x = std::hint::black_box(f64::from_bits(a));
    let y = std::hint::black_box(f64::from_bits(b));
    let z = std::hint::black_box(f64::from_bits(c));
    std::hint::black_box(x.mul_add(y, z)).to_bits()
}

// --- Conversions ----------------------------------------------------------

#[must_use]
pub fn i64_to_f64(x: i64, mode: RoundingMode) -> u64 {
    let _g = rmode::Guard::new(mode);
    // black_box on input so LLVM cannot hoist the FPCR-sensitive cast
    // past the Guard's `msr fpcr, ...`. Same pattern as cpu::fadd.
    let x = std::hint::black_box(x);
    std::hint::black_box(x as f64).to_bits()
}

#[must_use]
pub fn u64_to_f64(x: u64, mode: RoundingMode) -> u64 {
    let _g = rmode::Guard::new(mode);
    let x = std::hint::black_box(x);
    std::hint::black_box(x as f64).to_bits()
}

/// f32 → f64 (exact; no rounding required, every f32 is representable).
#[must_use]
pub fn f32_to_f64(a: u32) -> u64 {
    f64::from(f32::from_bits(a)).to_bits()
}

#[must_use]
pub fn f64_to_f32(a: u64, mode: RoundingMode) -> u32 {
    let _g = rmode::Guard::new(mode);
    let x = std::hint::black_box(f64::from_bits(a));
    std::hint::black_box(x as f32).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(x: f64) -> u64 {
        x.to_bits()
    }

    #[test]
    fn fadd_basic() {
        assert_eq!(fadd(bits(1.0), bits(2.0), RoundingMode::Nearest), bits(3.0));
        assert_eq!(fadd(bits(-1.5), bits(0.25), RoundingMode::Nearest), bits(-1.25));
    }

    #[test]
    fn fsub_basic() {
        assert_eq!(fsub(bits(5.0), bits(3.0), RoundingMode::Nearest), bits(2.0));
    }

    #[test]
    fn fmul_basic() {
        assert_eq!(fmul(bits(2.5), bits(4.0), RoundingMode::Nearest), bits(10.0));
    }

    #[test]
    fn fdiv_basic() {
        assert_eq!(fdiv(bits(1.0), bits(2.0), RoundingMode::Nearest), bits(0.5));
    }

    #[test]
    fn fsqrt_basic() {
        assert_eq!(fsqrt(bits(4.0), RoundingMode::Nearest), bits(2.0));
        assert_eq!(fsqrt(bits(2.0), RoundingMode::Nearest), bits(2.0f64.sqrt()));
    }

    #[test]
    fn cvt_i32_cases() {
        assert_eq!(cvt_i32_to_f64(0), bits(0.0));
        assert_eq!(cvt_i32_to_f64(1), bits(1.0));
        assert_eq!(cvt_i32_to_f64(-1), bits(-1.0));
        assert_eq!(cvt_i32_to_f64(42), bits(42.0));
        assert_eq!(cvt_i32_to_f64(i32::MIN), bits(-2_147_483_648.0));
        assert_eq!(cvt_i32_to_f64(i32::MAX), bits(2_147_483_647.0));
    }

    /// 1/3 is the classic "every rounding mode gives a different bit
    /// pattern" case. Exact value sits between two representables.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rounding_modes_diverge_on_one_third() {
        let a = bits(1.0);
        let b = bits(3.0);
        let near = fdiv(a, b, RoundingMode::Nearest);
        let down = fdiv(a, b, RoundingMode::Down);
        let up = fdiv(a, b, RoundingMode::Up);
        let zero = fdiv(a, b, RoundingMode::Zero);

        assert_eq!(near, down); // 1/3 rounds down to the nearest even
        assert_eq!(down, zero); // positive result → zero == down
        assert_ne!(down, up); // up flips the last bit
        assert_eq!(up, down + 1);
    }

    /// FADD with opposite-signed addends of different magnitudes —
    /// exercises the "subtract-then-normalize" branch of hardware FADD.
    #[test]
    fn fadd_cancellation_precision() {
        let a = bits(1.0e16);
        let b = bits(-1.0e16 + 1.0); // approximately 1.0 after cancellation
        let r = fadd(a, b, RoundingMode::Nearest);
        // Exact match with native f64 is the requirement here.
        assert_eq!(r, (1.0e16f64 + (-1.0e16f64 + 1.0)).to_bits());
    }
}
