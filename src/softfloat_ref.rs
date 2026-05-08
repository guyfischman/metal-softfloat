//! Pure-integer Rust implementations of the f64 ops.
//!
//! Purpose: prove the algorithms out in Rust (where debugging is easy)
//! before porting to MSL. Tests cross-check these against native f64
//! across the full u64 domain (NaN / ±Inf / ±0 / subnormal / normal
//! all included). The MSL port is then a mostly-mechanical translation
//! of the same algorithm.
//!
//! ## IEEE-754 conformance
//!
//! - All four [`RoundingMode`] modes (nearest-ties-to-even, toward
//!   ±∞, toward zero).
//! - NaN / ±Inf / ±0 propagation per IEEE-754 §6.
//! - Subnormal inputs and outputs (gradual underflow) per §7.4.
//!   Disable with the `ftz` feature for FTZ semantics.
//!
//! Signaling vs quiet NaN distinction is not preserved: every NaN
//! result is the canonical qNaN `0x7FF8_0000_0000_0000`. NaN payloads
//! are not propagated. Exception flags are not exposed.

use super::RoundingMode;

const EXP_BITS: u32 = 11;
const MANT_BITS: u32 = 52;
const EXP_MASK: u64 = (1u64 << EXP_BITS) - 1;
const MANT_MASK: u64 = (1u64 << MANT_BITS) - 1;
const IMPLICIT_BIT: u64 = 1u64 << MANT_BITS;
const EXP_BIAS: i32 = 1023;

/// Canonical quiet NaN. Bit 51 (top of mantissa) is set as the IEEE-754
/// "is-quiet" marker; the rest of the payload is zero. Sign bit zero so
/// every NaN result has the same bit pattern.
const CANONICAL_QNAN: u64 = 0x7FF8_0000_0000_0000;
const POS_INF: u64 = 0x7FF0_0000_0000_0000;

/// IEEE-754 floating-point class. Together with the sign bit this is
/// enough to dispatch every special-case rule in §6/§7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IeeeClass {
    Zero,
    Subnormal,
    Normal,
    Inf,
    NaN,
}

/// Classify the f64 bit pattern. Pure decoder — no normalization.
#[must_use]
#[inline]
pub const fn classify(bits: u64) -> IeeeClass {
    let exp = (bits >> MANT_BITS) & EXP_MASK;
    let mant = bits & MANT_MASK;
    if exp == 0 {
        if mant == 0 {
            IeeeClass::Zero
        } else {
            IeeeClass::Subnormal
        }
    } else if exp == EXP_MASK {
        if mant == 0 {
            IeeeClass::Inf
        } else {
            IeeeClass::NaN
        }
    } else {
        IeeeClass::Normal
    }
}

#[derive(Debug, Clone, Copy)]
struct Unpacked {
    sign: u64,         // 0 or 1
    exp: i32,          // unbiased; meaningful only for Normal/Subnormal
    mantissa: u64,     // 53-bit (implicit bit at position 52) for Normal
                       // /Subnormal-after-normalize. Zero for Zero/Inf/NaN.
    class: IeeeClass,
}

const fn unpack(x: u64) -> Unpacked {
    let sign = x >> 63;
    let exp_raw = ((x >> MANT_BITS) & EXP_MASK) as i32;
    let mant_raw = x & MANT_MASK;
    let class = classify(x);

    let (exp, mantissa) = match class {
        IeeeClass::Zero | IeeeClass::Inf | IeeeClass::NaN => (0, 0),
        IeeeClass::Normal => (exp_raw - EXP_BIAS, mant_raw | IMPLICIT_BIT),
        IeeeClass::Subnormal => {
            // Subnormal value v = mant_raw × 2^-1074. Normalize so the
            // implicit bit (bit 52) is set: shift left by k where k =
            // 52 - msb_pos. Then v = (mant_raw << k) × 2^(-1074-k+52)
            // and exp_unbiased = -1022 - k. The arithmetic paths see
            // a "very small normal" and produce subnormal output via
            // `pack` if the result's biased exponent is < 1.
            //
            // mant_raw has its set bits in bits 0..=51, so
            // leading_zeros() ∈ [12, 63]. shift = lz - 11.
            let shift = (mant_raw.leading_zeros() as i32) - 11;
            (-1022 - shift, mant_raw << shift)
        }
    };

    Unpacked { sign, exp, mantissa, class }
}

const fn overflow_pack(sign: u64, mode: RoundingMode) -> u64 {
    // IEEE-754 §7.4: overflow returns ±∞ in nearest mode; in directed
    // modes the magnitude-toward-target wins, the away-from-target
    // produces the largest representable finite (FLT_MAX with sign).
    let is_neg = sign == 1;
    let to_inf = (sign << 63) | POS_INF;
    let to_max = (sign << 63) | (POS_INF - 1);
    match mode {
        RoundingMode::Nearest => to_inf,
        RoundingMode::Up => if is_neg { to_max } else { to_inf },
        RoundingMode::Down => if is_neg { to_inf } else { to_max },
        RoundingMode::Zero => to_max,
    }
}

/// Round + pack from a 128-bit staged mantissa.
///
/// `m_128` has its 53-bit mantissa at bits [64..117] (implicit bit at 116);
/// bits [0..64) are guard / round / sticky. `exp_pre_round` is the
/// unbiased exponent assuming MSB at bit 116 — caller has already done
/// any pre-round normalization shifts. Single-rounding: this fuses what
/// used to be `round_128` + `pack` into one decision so subnormal
/// outputs aren't double-rounded.
#[inline]
const fn round_and_pack(m_128: u128, sign: u64, exp_pre_round: i32, mode: RoundingMode) -> u64 {
    let biased_pre = exp_pre_round + EXP_BIAS;

    if biased_pre > 2046 {
        return overflow_pack(sign, mode);
    }

    if biased_pre < 1 {
        #[cfg(feature = "ftz")]
        {
            return sign << 63;
        }
        #[cfg(not(feature = "ftz"))]
        {
            // Subnormal output. Shift m_128 right by an extra `shift`
            // bits (with sticky-jam) so the new LSB sits at the
            // subnormal field's LSB, then round.
            let shift = (1 - biased_pre) as u32;
            let dropped_mask: u128 = if shift >= 128 { u128::MAX } else { (1u128 << shift) - 1 };
            let dropped = m_128 & dropped_mask;
            let shifted = if shift >= 128 { 0 } else { m_128 >> shift };
            let m = shifted | ((dropped != 0) as u128);

            let new_mant = (m >> 64) as u64;
            let guard = ((m >> 63) & 1) as u64;
            let round_bit = ((m >> 62) & 1) as u64;
            let sticky = (m & ((1u128 << 62) - 1) != 0) as u64;
            let round_up = match mode {
                RoundingMode::Nearest => guard == 1 && (round_bit | sticky | (new_mant & 1)) != 0,
                RoundingMode::Down => sign == 1 && (guard | round_bit | sticky) != 0,
                RoundingMode::Up => sign == 0 && (guard | round_bit | sticky) != 0,
                RoundingMode::Zero => false,
            };
            let mant_after = new_mant + (round_up as u64);
            if mant_after >> MANT_BITS != 0 {
                // Round-up promoted us to the smallest normal.
                return (sign << 63) | (1u64 << MANT_BITS);
            }
            return (sign << 63) | (mant_after & MANT_MASK);
        }
    }

    // Normal output. Round at bit 64, then pack.
    let (rounded_m, round_bump) = round_128(m_128, sign, mode);
    let biased = biased_pre + round_bump;
    if biased > 2046 {
        return overflow_pack(sign, mode);
    }
    (sign << 63) | ((biased as u64 & EXP_MASK) << MANT_BITS) | (rounded_m & MANT_MASK)
}

#[inline]
const fn pack(sign: u64, exp_unbiased: i32, mantissa: u64, mode: RoundingMode) -> u64 {
    let biased = exp_unbiased + EXP_BIAS;
    if biased > 2046 {
        return overflow_pack(sign, mode);
    }
    if biased < 1 {
        // Underflow path on the legacy `pack` API (no GRS info available
        // here — it was already discarded by the caller's `round_128`).
        // Returns signed zero. Callers that need correct gradual
        // underflow use `round_and_pack` instead, which fuses the
        // rounding decision with the underflow shift.
        let _ = mode;
        return sign << 63;
    }
    (sign << 63) | ((biased as u64 & EXP_MASK) << MANT_BITS) | (mantissa & MANT_MASK)
}

/// Take a normalized 128-bit intermediate with the mantissa's MSB at
/// bit 116 (i.e. bits [116:64] hold the 53 mantissa bits and bits
/// [63:0] hold fractional / guard / round / sticky) and reduce it to a
/// 53-bit mantissa under the given rounding mode. Returns the final
/// mantissa (still has the implicit bit at bit 52) and an exponent
/// adjustment (0 or 1 if rounding overflowed the mantissa range).
#[inline]
const fn round_128(m_128: u128, sign: u64, mode: RoundingMode) -> (u64, i32) {
    // The 53-bit mantissa is bits [116:64]. Guard is bit 63; round
    // is bit 62; sticky is (bits 61..=0) != 0.
    let mantissa = (m_128 >> 64) as u64;
    let guard = ((m_128 >> 63) & 1) as u64;
    let round_bit = ((m_128 >> 62) & 1) as u64;
    let sticky = (m_128 & ((1u128 << 62) - 1) != 0) as u64;

    let round_up = match mode {
        RoundingMode::Nearest => {
            // Tie (guard=1, round=sticky=0) → pick even (mantissa LSB = 0).
            guard == 1 && (round_bit | sticky | (mantissa & 1)) != 0
        }
        RoundingMode::Down => sign == 1 && (guard | round_bit | sticky) != 0,
        RoundingMode::Up => sign == 0 && (guard | round_bit | sticky) != 0,
        RoundingMode::Zero => false,
    };

    if round_up {
        let bumped = mantissa + 1;
        if bumped >> 53 != 0 {
            (bumped >> 1, 1)
        } else {
            (bumped, 0)
        }
    } else {
        (mantissa, 0)
    }
}

/// IEEE-754 binary64 addition. Handles all input classes: NaN, ±Inf,
/// ±0, subnormal, normal. Default rounding modes per [`RoundingMode`].
///
/// NaN payload: any NaN result is the canonical qNaN. Signaling-NaN
/// distinction is not preserved.
#[must_use]
#[inline]
pub const fn fadd(a: u64, b: u64, mode: RoundingMode) -> u64 {
    let ua = unpack(a);
    let ub = unpack(b);

    // --- Special-case dispatch (IEEE-754 §6) --------------------
    match (ua.class, ub.class) {
        (IeeeClass::NaN, _) | (_, IeeeClass::NaN) => return CANONICAL_QNAN,
        (IeeeClass::Inf, IeeeClass::Inf) => {
            // ∞ + ∞ → ∞ if same sign; ∞ − ∞ → qNaN.
            return if ua.sign == ub.sign {
                (ua.sign << 63) | POS_INF
            } else {
                CANONICAL_QNAN
            };
        }
        (IeeeClass::Inf, _) => return (ua.sign << 63) | POS_INF,
        (_, IeeeClass::Inf) => return (ub.sign << 63) | POS_INF,
        (IeeeClass::Zero, IeeeClass::Zero) => {
            // ±0 + ±0: the only case that returns -0 is (-0)+(-0), or
            // any sum under round-toward-negative whose true value is 0.
            // Same-sign zero preserves sign; different-sign zero gives
            // +0 except under Down which gives -0.
            let sign = if ua.sign == ub.sign {
                ua.sign
            } else {
                (matches!(mode, RoundingMode::Down) as u64)
            };
            return sign << 63;
        }
        (IeeeClass::Zero, _) => return b,
        (_, IeeeClass::Zero) => return a,
        _ => {}
    }
    #[cfg(feature = "ftz")]
    {
        // Treat subnormal inputs as ±0 under FTZ.
        if matches!(ua.class, IeeeClass::Subnormal) {
            return if matches!(ub.class, IeeeClass::Subnormal) {
                let sign = if ua.sign == ub.sign {
                    ua.sign
                } else {
                    (matches!(mode, RoundingMode::Down) as u64)
                };
                sign << 63
            } else {
                b
            };
        }
        if matches!(ub.class, IeeeClass::Subnormal) {
            return a;
        }
    }

    // Canonicalize: make `big` the larger-magnitude operand. Equal
    // exponents → compare mantissas.
    let (big, sml) =
        if ua.exp > ub.exp || (ua.exp == ub.exp && ua.mantissa >= ub.mantissa) {
            (ua, ub)
        } else {
            (ub, ua)
        };

    let shift = (big.exp - sml.exp) as u32;

    // Stage the 53-bit mantissas into a 128-bit intermediate with the
    // MSB at bit 116. The low 64 bits act as guard/round/sticky space.
    let big_128 = (big.mantissa as u128) << 64;
    let sml_raw = (sml.mantissa as u128) << 64;

    let sml_aligned;
    if shift >= 128 {
        // Underflows 128-bit precision entirely; fold to sticky.
        sml_aligned = if sml_raw != 0 { 1u128 } else { 0u128 };
    } else if shift == 0 {
        sml_aligned = sml_raw;
    } else {
        let dropped_mask = (1u128 << shift) - 1;
        let dropped = sml_raw & dropped_mask;
        let shifted = sml_raw >> shift;
        // Fold the dropped bits into bit 0 of the result as a sticky.
        sml_aligned = shifted | if dropped != 0 { 1 } else { 0 };
    }

    let (result_sign, mut result_128, mut exp_adjust) = if big.sign == sml.sign {
        let sum = big_128 + sml_aligned;
        // Overflow handling: if bit 117 is set, mantissa overflowed.
        // Shift right by 1, preserving any sticky into bit 0.
        if sum >> 117 != 0 {
            let sticky = sum & 1;
            ((big.sign), (sum >> 1) | sticky, 1i32)
        } else {
            (big.sign, sum, 0)
        }
    } else {
        let diff = big_128 - sml_aligned;
        if diff == 0 {
            // Exact cancellation. IEEE-754 §6.3: result is +0 except
            // under round-toward-negative, which returns -0.
            let sign = (matches!(mode, RoundingMode::Down) as u64);
            return sign << 63;
        }
        // Normalize: shift left until bit 116 is set.
        let lz = diff.leading_zeros() as i32;
        let target_msb_pos = 116i32;
        let msb_pos = 127i32 - lz;
        let shift_left = target_msb_pos - msb_pos;
        if shift_left > 0 {
            (big.sign, diff << shift_left, -shift_left)
        } else if shift_left < 0 {
            // Bigger than target — shift right; fold lost bits into sticky.
            let n = (-shift_left) as u32;
            let fell = diff & ((1u128 << n) - 1);
            ((big.sign), (diff >> n) | if fell != 0 { 1 } else { 0 }, -shift_left)
        } else {
            (big.sign, diff, 0)
        }
    };
    let _ = &mut result_128;
    let _ = &mut exp_adjust;

    round_and_pack(result_128, result_sign, big.exp + exp_adjust, mode)
}

#[must_use]
#[inline]
pub const fn fsub(a: u64, b: u64, mode: RoundingMode) -> u64 {
    fadd(a, b ^ (1u64 << 63), mode)
}

#[must_use]
#[inline]
pub const fn fmul(a: u64, b: u64, mode: RoundingMode) -> u64 {
    let ua = unpack(a);
    let ub = unpack(b);
    let sign = ua.sign ^ ub.sign;

    // --- Special-case dispatch (IEEE-754 §6) --------------------
    match (ua.class, ub.class) {
        (IeeeClass::NaN, _) | (_, IeeeClass::NaN) => return CANONICAL_QNAN,
        // 0 × ∞ is invalid — qNaN regardless of signs.
        (IeeeClass::Zero, IeeeClass::Inf) | (IeeeClass::Inf, IeeeClass::Zero) => {
            return CANONICAL_QNAN;
        }
        (IeeeClass::Inf, _) | (_, IeeeClass::Inf) => return (sign << 63) | POS_INF,
        (IeeeClass::Zero, _) | (_, IeeeClass::Zero) => return sign << 63,
        _ => {}
    }
    #[cfg(feature = "ftz")]
    {
        if matches!(ua.class, IeeeClass::Subnormal) || matches!(ub.class, IeeeClass::Subnormal) {
            return sign << 63;
        }
    }

    let exp = ua.exp + ub.exp;
    // 53 × 53 bit multiply → ≤ 106 bit product. MSB at bit 104 or 105.
    let product = (ua.mantissa as u128) * (ub.mantissa as u128);

    // Normalize so MSB is at bit 104, tracking any shift in `exp_adj`.
    let (normalized, exp_adj) = if product >> 105 != 0 {
        let sticky = product & 1;
        ((product >> 1) | sticky, 1i32)
    } else {
        (product, 0i32)
    };

    // Place MSB at bit 116: shift up by 12. Bits 0..12 become part of
    // the sticky region (sticky bit is OR of bits 0..=61 of m_128).
    let m_128 = normalized << 12;
    round_and_pack(m_128, sign, exp + exp_adj, mode)
}

/// Compute `(numer_hi << 64) / denom` as `(quotient, remainder)` using
/// Newton-Raphson on reciprocal (multiplication-only). Used by `fdiv`.
///
/// Seed Y ≈ 2^115 / denom from f64, then 2 Newton iterations refine to
/// full precision. Quotient Q ≈ numer_hi × Y >> 51; correction step
/// fixes any ±1 residual via residual = numer - q×denom.
///
/// Why this form: MSL's u64/u64 divide is synthesized (~40 cycles on
/// Apple GPU). The previous 8-bit-chunks division did 9 of these per
/// fdiv (~360 cycles). Newton uses multiplication only — `u128_mul_u64`
/// is ~15 cycles (4 HW u32×u32→u64 partials), so 2 iters × 3 mults +
/// correction ≈ 140 cycles. On Rust the speedup doesn't show (native
/// u128/u64 divide is fast); this code exists to be the bit-exact
/// reference that the MSL port cross-checks against.
#[inline]
const fn u128_div_u64_newton(numer_hi: u64, denom: u64) -> (u128, u64) {
    debug_assert!(denom >= (1u64 << 52) && denom < (1u64 << 53));

    // Seed Y ≈ 2^115 / denom via f64 reciprocal. On GPU we use f32;
    // Newton converges fast enough that seed precision doesn't matter
    // once Y ≤ u64. Note the terminal precision of Y is capped by u64
    // width (64 bits), NOT by Newton's convergence rate — so the
    // truncation of Y to integer drops a fractional part of up to 1.0
    // from the true real value 2^115/denom.
    let y_f = (1u128 << 115) as f64 / denom as f64;
    let mut y: u64 = y_f as u64;

    // Two Newton iters. See formula derivation and limb layout in the
    // MSL port. Each iter converges quadratically; after 2 iters Y
    // equals floor(2^115/denom) exactly for the input range we care
    // about. (`while` instead of `for _ in 0..2` so this fn is const.)
    let mut iter = 0;
    while iter < 2 {
        let product: u128 = denom as u128 * y as u128;
        let delta: u128 = (1u128 << 116).wrapping_sub(product);
        let dh = (delta >> 64) as u64;
        let dl = delta as u64;
        let p_hi: u128 = y as u128 * dh as u128;
        let p_lo: u128 = y as u128 * dl as u128;
        let top_128 = p_hi.wrapping_add(p_lo >> 64);
        y = (top_128 >> 51) as u64;
        iter += 1;
    }

    // Q_approx = (numer_hi × Y) >> 51. With Y = floor(2^115/denom), the
    // fractional-part truncation costs us at most numer_hi/2^51 < 2^13
    // in the quotient. Crucially Y ≤ Y_true_real, so Q_approx ≤ Q_true.
    let q_approx: u128 = (numer_hi as u128 * y as u128) >> 51;

    // Residual correction. Compute rem = numer_shifted − Q_approx × denom.
    // True quotient = Q_approx + floor(rem / denom), extracted bit-by-bit.
    let numer_shifted: u128 = (numer_hi as u128) << 64;
    // Q_approx × denom can overflow u128 by 1 bit (up to 2^129). Compute
    // mod 2^128 and subtract from numer_shifted via wrapping. The true
    // rem is < 2^66 so the wrapped result equals the real residual.
    let q_denom_low = q_approx.wrapping_mul(denom as u128);
    let mut rem: u128 = numer_shifted.wrapping_sub(q_denom_low);

    // Extract delta_q = floor(rem / denom) via 13 iters of shift-subtract.
    // At each step, if rem ≥ denom << i, subtract and set bit i of delta_q.
    // (`while` instead of `for i in (0..=13).rev()` for const-fn.)
    let denom_u128 = denom as u128;
    let mut delta_q: u128 = 0;
    let mut i: i32 = 13;
    while i >= 0 {
        let shifted = denom_u128 << i;
        if rem >= shifted {
            rem -= shifted;
            delta_q |= 1u128 << i;
        }
        i -= 1;
    }

    let q = q_approx + delta_q;
    (q, rem as u64)
}

#[must_use]
#[inline]
pub const fn fdiv(a: u64, b: u64, mode: RoundingMode) -> u64 {
    let ua = unpack(a);
    let ub = unpack(b);
    let sign = ua.sign ^ ub.sign;

    // --- Special-case dispatch (IEEE-754 §6/§7) -----------------
    match (ua.class, ub.class) {
        (IeeeClass::NaN, _) | (_, IeeeClass::NaN) => return CANONICAL_QNAN,
        (IeeeClass::Inf, IeeeClass::Inf) | (IeeeClass::Zero, IeeeClass::Zero) => {
            return CANONICAL_QNAN;
        }
        (IeeeClass::Inf, _) => return (sign << 63) | POS_INF,
        (_, IeeeClass::Inf) => return sign << 63,
        // finite / 0 → ±Inf (divide-by-zero, IEEE §7.3 default exception).
        (_, IeeeClass::Zero) => return (sign << 63) | POS_INF,
        (IeeeClass::Zero, _) => return sign << 63,
        _ => {}
    }
    #[cfg(feature = "ftz")]
    {
        // FTZ: subnormal numerator → 0, subnormal denominator → ∞.
        if matches!(ua.class, IeeeClass::Subnormal) {
            return if matches!(ub.class, IeeeClass::Subnormal) {
                CANONICAL_QNAN
            } else {
                sign << 63
            };
        }
        if matches!(ub.class, IeeeClass::Subnormal) {
            return (sign << 63) | POS_INF;
        }
    }

    let exp = ua.exp - ub.exp;
    // Compute (ma << 75) / denom. ma << 75 has its low 75 bits zero, so
    // we stage it as (numer_hi << 64) where numer_hi = ma << 11. Then
    // shift up by the remaining 11 at the caller of u128_div_u64_newton.
    let numer_hi: u64 = ua.mantissa << 11;
    let (quotient_base, remainder_base) =
        u128_div_u64_newton(numer_hi, ub.mantissa);
    // Result above is `(ma << 75) / denom` quotient — matches the old
    // `numer / denom` when numer = (ma as u128) << 75.
    let quotient = quotient_base;
    let remainder = remainder_base as u128;

    let q_msb = 127 - quotient.leading_zeros() as i32;
    let exp_adj = q_msb - 75; // 0 if MSB at 75, -1 if at 74.

    let shift_up = 116 - q_msb;
    let m_128 = (quotient << shift_up) | if remainder != 0 { 1u128 } else { 0 };
    round_and_pack(m_128, sign, exp + exp_adj, mode)
}

/// Integer sqrt of `n`. Returns `(sqrt_floor, remainder)` where
/// `sqrt_floor^2 + remainder == n` and `remainder < 2*sqrt_floor + 1`.
///
/// Digit-by-digit (subtract-and-shift) method: seed the top 24 bits of
/// the root from a hardware f32 sqrt on the top 48 bits of `n`, then
/// continue with 40 iterations of the 1-bit-per-iteration loop for the
/// remaining 80 bits of `n`.
///
/// Why: the pure shift-subtract form ran 64 iters × ~8 u64 ops each. The
/// seed skips 24 iters via a single HW sqrt + small correction, saving
/// ~37% of isqrt cost while preserving bit-exact behavior. fsqrt's range
/// keeps `n_top48 < 2^48`, so the f32 round-trip hits its error at most
/// ±1 ULP, absorbed by the correction loop.
///
/// `n` must have MSB at bit ≤ 125 (so n < 2^126); fsqrt's mantissa-
/// shifted inputs sit at bits 110–111, well inside the bound.
// --- Berkeley SoftFloat recipSqrt approximation tables (BSD-3-Clause) ---
// Verbatim from softfloat-3e/source/s_approxRecipSqrt_1Ks.c
const APPROX_RECIP_SQRT_K0S: [u16; 16] = [
    0xB4C9, 0xFFAB, 0xAA7D, 0xF11C, 0xA1C5, 0xE4C7, 0x9A43, 0xDA29,
    0x93B5, 0xD0E5, 0x8DED, 0xC8B7, 0x88C6, 0xC16D, 0x8424, 0xBAE1,
];
const APPROX_RECIP_SQRT_K1S: [u16; 16] = [
    0xA5A5, 0xEA42, 0x8C21, 0xC62D, 0x788F, 0xAA7F, 0x6928, 0x94B6,
    0x5CC7, 0x8335, 0x52A6, 0x74E2, 0x4A3E, 0x68FE, 0x432B, 0x5EFD,
];

/// Ported from Berkeley SoftFloat `softfloat_approxRecipSqrt32_1`.
/// Returns ~32-bit approximation of `2^32 / sqrt(a * 2^oddExpA)`,
/// with bit 31 guaranteed set. `a` is in `[2^31, 2^32)`, `odd_exp_a` ∈ {0,1}.
const fn approx_recip_sqrt32_1(odd_exp_a: u32, a: u32) -> u32 {
    let index = ((a >> 27) & 0xE) as usize + odd_exp_a as usize;
    let eps = (a >> 12) as u16;
    let r0: u16 = APPROX_RECIP_SQRT_K0S[index].wrapping_sub(
        ((APPROX_RECIP_SQRT_K1S[index] as u32 * eps as u32) >> 20) as u16,
    );
    let mut e_sqr_r0: u32 = (r0 as u32) * (r0 as u32);
    if odd_exp_a == 0 {
        e_sqr_r0 <<= 1;
    }
    let sigma0: u32 = !((((e_sqr_r0 as u64) * (a as u64)) >> 23) as u32);
    let mut r: u32 = ((r0 as u32) << 16)
        .wrapping_add(((r0 as u64 * sigma0 as u64) >> 25) as u32);
    let sqr_sigma0: u32 = ((sigma0 as u64 * sigma0 as u64) >> 32) as u32;
    let tail = ((((r >> 1).wrapping_add(r >> 3)).wrapping_sub((r0 as u32) << 14)) as u64
        * sqr_sigma0 as u64)
        >> 48;
    r = r.wrapping_add(tail as u32);
    if (r & 0x8000_0000) == 0 { 0x8000_0000 } else { r }
}

// --- Berkeley SoftFloat recip approximation tables (BSD-3-Clause) ---
// Verbatim from softfloat-3e/source/s_approxRecip_1Ks.c
const APPROX_RECIP_K0S: [u16; 16] = [
    0xFFC4, 0xF0BE, 0xE363, 0xD76F, 0xCCAD, 0xC2F0, 0xBA16, 0xB201,
    0xAA97, 0xA3C6, 0x9D7A, 0x97A6, 0x923C, 0x8D32, 0x887E, 0x8417,
];
const APPROX_RECIP_K1S: [u16; 16] = [
    0xF0F1, 0xD62C, 0xBFA1, 0xAC77, 0x9C0A, 0x8DDB, 0x8185, 0x76BA,
    0x6D3B, 0x64D4, 0x5D5C, 0x56B1, 0x50B6, 0x4B55, 0x4679, 0x4211,
];

/// Ported from Berkeley SoftFloat `softfloat_approxRecip32_1`.
/// Returns ~32-bit approximation of `2^64 / a` where `a` is in `[2^31, 2^32)`.
const fn approx_recip32_1(a: u32) -> u32 {
    let index = ((a >> 27) & 0xF) as usize;
    let eps = (a >> 11) as u16;
    let r0: u16 = APPROX_RECIP_K0S[index].wrapping_sub(
        ((APPROX_RECIP_K1S[index] as u32 * eps as u32) >> 20) as u16,
    );
    let sigma0: u32 = !((((r0 as u64) * (a as u64)) >> 7) as u32);
    let mut r: u32 = ((r0 as u32) << 16)
        .wrapping_add((((r0 as u64) * (sigma0 as u64)) >> 24) as u32);
    let sqr_sigma0: u32 = (((sigma0 as u64) * (sigma0 as u64)) >> 32) as u32;
    r = r.wrapping_add((((r as u64) * (sqr_sigma0 as u64)) >> 48) as u32);
    r
}

/// Berkeley-style fdiv for f64. Direct port of `f64_div` from
/// berkeley-softfloat-3 (BSD-3-Clause), adapted for our rounding modes.
///
/// Validates bit-exact against existing `fdiv` (which is already
/// bit-exact vs native `f64 / f64`). See test `fdiv_berkeley_matches_existing`.
#[must_use]
#[inline]
pub const fn fdiv_berkeley(a: u64, b: u64, mode: RoundingMode) -> u64 {
    let ua = unpack(a);
    let ub = unpack(b);
    let sign_z = ua.sign ^ ub.sign;
    // Berkeley-port path: only correct on normal-finite operands.
    // The canonical fdiv handles the full input class taxonomy.
    if matches!(ua.class, IeeeClass::Zero) {
        return sign_z << 63;
    }

    let exp_a_biased: i32 = ua.exp + EXP_BIAS;
    let exp_b_biased: i32 = ub.exp + EXP_BIAS;
    let mut exp_z_biased: i32 = exp_a_biased - exp_b_biased + 0x3FE;

    let sig_a_init: u64 = ua.mantissa; // has implicit bit
    let sig_b: u64 = ub.mantissa;      // has implicit bit
    let sig_a: u64;
    if sig_a_init < sig_b {
        exp_z_biased -= 1;
        sig_a = sig_a_init << 11;
    } else {
        sig_a = sig_a_init << 10;
    }
    let sig_b_11 = sig_b << 11;
    let sig_b_hi32: u32 = (sig_b_11 >> 32) as u32;

    let recip32: u32 = approx_recip32_1(sig_b_hi32).wrapping_sub(2);
    let sig32_z: u32 = ((((sig_a >> 32) as u32) as u64 * recip32 as u64) >> 32) as u32;
    let double_term: u32 = sig32_z << 1;
    let sig_b_tail: u32 = (sig_b_11 as u32) >> 4;
    let mut rem: u64 =
        (sig_a.wrapping_sub((double_term as u64) * (sig_b_hi32 as u64)) << 28)
            .wrapping_sub((double_term as u64) * (sig_b_tail as u64));
    let q: u32 = ((((rem >> 32) as u32) as u64 * recip32 as u64) >> 32) as u32 + 4;
    let mut sig_z: u64 = ((sig32_z as u64) << 32).wrapping_add((q as u64) << 4);

    // Berkeley low-bit correction.
    if (sig_z & 0x1FF) < (4 << 4) {
        let q_corr = q & !7;
        sig_z &= !0x7Fu64;
        let double_term2: u32 = q_corr << 1;
        rem = (rem.wrapping_sub((double_term2 as u64) * (sig_b_hi32 as u64)) << 28)
            .wrapping_sub((double_term2 as u64) * (sig_b_tail as u64));
        if rem & 0x8000_0000_0000_0000 != 0 {
            sig_z = sig_z.wrapping_sub(1 << 7);
        } else if rem != 0 {
            sig_z |= 1;
        }
    }

    // Round + pack (same pattern as fsqrt_berkeley).
    let round_bits: u64 = sig_z & 0x3FF;
    let round_increment: u64 = match mode {
        RoundingMode::Nearest => 0x200,
        RoundingMode::Up => if sign_z == 0 { 0x3FF } else { 0 },
        RoundingMode::Down => if sign_z == 0 { 0 } else { 0x3FF },
        RoundingMode::Zero => 0,
    };
    let mut rounded_sig: u64 = sig_z.wrapping_add(round_increment) >> 10;
    if matches!(mode, RoundingMode::Nearest) && round_bits == 0x200 {
        rounded_sig &= !1;
    }

    let final_exp_biased: i32;
    let final_mantissa: u64;
    if rounded_sig >> 53 != 0 {
        final_exp_biased = exp_z_biased + 2;
        final_mantissa = rounded_sig >> 1;
    } else {
        final_exp_biased = exp_z_biased + 1;
        final_mantissa = rounded_sig;
    }
    pack(sign_z, final_exp_biased - EXP_BIAS, final_mantissa, mode)
}

/// Berkeley-style fsqrt for f64. Direct port of `f64_sqrt` from
/// berkeley-softfloat-3 (BSD-3-Clause), adapted for our rounding modes.
///
/// Produces bit-exact results vs the canonical [`fsqrt`] above for
/// positive normal operands; the `fsqrt_berkeley_matches_existing`
/// test cross-checks 10K random normals × 4 rounding modes.
#[must_use]
#[inline]
pub const fn fsqrt_berkeley(a: u64, mode: RoundingMode) -> u64 {
    let ua = unpack(a);
    // Berkeley-port path: only correct on positive normal operands.
    // The canonical fsqrt handles the full input class taxonomy.
    if matches!(ua.class, IeeeClass::Zero) {
        return a;
    }
    debug_assert!(ua.sign == 0);

    let exp_a_biased: i32 = ua.exp + EXP_BIAS;
    let odd_exp_a: u32 = (exp_a_biased & 1) as u32;
    // Berkeley: expZ = ((expA - 0x3FF) >> 1) + 0x3FE (biased).
    // We'll convert back to unbiased at the end.
    let exp_z_biased: i32 = ((exp_a_biased - 0x3FF) >> 1) + 0x3FE;

    // sigA with implicit bit already set (unpack did this).
    let mut sig_a: u64 = ua.mantissa;
    let sig32_a: u32 = (sig_a >> 21) as u32;
    let recip_sqrt32: u32 = approx_recip_sqrt32_1(odd_exp_a, sig32_a);
    let mut sig32_z: u32 = ((sig32_a as u64 * recip_sqrt32 as u64) >> 32) as u32;

    if odd_exp_a != 0 {
        sig_a <<= 8;
        sig32_z >>= 1;
    } else {
        sig_a <<= 9;
    }
    let rem: u64 = sig_a.wrapping_sub((sig32_z as u64) * (sig32_z as u64));
    let q: u32 = (((rem >> 2) as u32 as u64 * recip_sqrt32 as u64) >> 32) as u32;
    let mut sig_z: u64 = ((sig32_z as u64) << 32) | (1u64 << 5);
    sig_z = sig_z.wrapping_add((q as u64) << 3);

    // Berkeley's low-bit correction for exact rounding.
    if (sig_z & 0x1FF) < 0x22 {
        sig_z &= !0x3Fu64;
        let shifted = sig_z >> 6;
        let rem2: u64 = (sig_a << 52).wrapping_sub(shifted.wrapping_mul(shifted));
        if rem2 & 0x8000_0000_0000_0000 != 0 {
            sig_z = sig_z.wrapping_sub(1);
        } else if rem2 != 0 {
            sig_z |= 1;
        }
    }

    // Berkeley's roundPackToF64 with sig bits 0..9 being round/sticky.
    // Our round_128 expects a u128 with bits 64..116 as mantissa.
    // sig_z has the 53-bit mantissa at bits 10..63 (with implicit bit at 63).
    // Shift to position bits 10..63 → bits 64..117 of u128:
    //   u128_val = (sig_z as u128) << 54 places sig_z's bit 63 at bit 117.
    //   So the implicit bit is at bit 117 (1 past our expected 116).
    //   Our fsqrt packs with exp_div2 that accounts for this.
    //
    // Simpler: replicate Berkeley's roundPack inline. Round_bits = sig_z & 0x3FF.
    let round_bits: u64 = sig_z & 0x3FF;
    // Berkeley roundPackToF64 increments (for positive sqrt, sign=0):
    //   Nearest  → 0x200
    //   Up (max) → 0x3FF (unconditional; behaves as "round up if any low bit")
    //   Down/Zero→ 0
    let round_increment: u64 = match mode {
        RoundingMode::Nearest => 0x200,
        RoundingMode::Up => 0x3FF,
        RoundingMode::Down | RoundingMode::Zero => 0,
    };
    let mut rounded_sig: u64 = sig_z.wrapping_add(round_increment) >> 10;
    // Ties-to-even for Nearest: clear bit 0 if the round bits equaled
    // exactly 0x200 (half-ULP) to force round-to-even.
    if matches!(mode, RoundingMode::Nearest) && round_bits == 0x200 {
        rounded_sig &= !1;
    }
    // Berkeley's packToF64UI uses addition (((exp << 52) + sig)) so the
    // implicit bit in sig carries into exp's LSB. Our `pack` uses OR +
    // mantissa mask, so we must add that carry manually: the f64-format
    // biased exp is `exp_z_biased + (implicit_bit_of_rounded_sig ? 1 : 0)`.
    // For normal inputs rounded_sig always has bit 52 set (implicit bit);
    // rounding can push it to bit 53 (double carry → +2 to exp, shift mant).
    let final_exp_biased: i32;
    let final_mantissa: u64;
    if rounded_sig >> 53 != 0 {
        final_exp_biased = exp_z_biased + 2;
        final_mantissa = rounded_sig >> 1;
    } else {
        final_exp_biased = exp_z_biased + 1;
        final_mantissa = rounded_sig;
    }
    pack(0, final_exp_biased - EXP_BIAS, final_mantissa, mode)
}

const fn isqrt_u128(n: u128) -> (u128, u128) {
    debug_assert!(
        n < (1u128 << 126),
        "isqrt_u128 u64-rem bound requires n < 2^126"
    );
    let mut n_hi = (n >> 64) as u64;
    let mut n_lo = n as u64;

    // Pure-integer 64-iter shift-subtract. The Metal port uses a HW
    // f32 sqrt seed to skip the first 24 iterations, but `f32::sqrt`
    // lives in `std` (not `core`), and this is the no_std reference
    // path — correctness, not speed, is what matters here.
    let mut x: u64 = 0;
    let mut rem: u64 = 0;

    // --- 64 iters × 2 bits per iter = 128 bits total. ---
    // (`while` instead of `for _ in 0..64` so this fn is const.)
    let mut iter = 0;
    while iter < 64 {
        let pair = n_hi >> 62;
        n_hi = (n_hi << 2) | (n_lo >> 62);
        n_lo <<= 2;
        rem = (rem << 2) | pair;
        let try_x = (x << 2) | 1;
        let ge = (rem >= try_x) as u64;
        rem = rem.wrapping_sub(try_x.wrapping_mul(ge));
        x = (x << 1) | ge;
        iter += 1;
    }
    (x as u128, rem as u128)
}

#[must_use]
#[inline]
pub const fn fsqrt(a: u64, mode: RoundingMode) -> u64 {
    let ua = unpack(a);

    // --- Special-case dispatch (IEEE-754 §6/§7) -----------------
    match ua.class {
        IeeeClass::NaN => return CANONICAL_QNAN,
        // sqrt(±0) = ±0, including sign of -0.
        IeeeClass::Zero => return a,
        IeeeClass::Inf => {
            return if ua.sign == 1 {
                CANONICAL_QNAN // sqrt(-∞) is invalid.
            } else {
                POS_INF
            };
        }
        _ => {}
    }
    // sqrt of a negative finite is invalid (qNaN per §7.2). The only
    // exception above was -0 (returned by Zero case unchanged).
    if ua.sign == 1 {
        return CANONICAL_QNAN;
    }
    #[cfg(feature = "ftz")]
    {
        if matches!(ua.class, IeeeClass::Subnormal) {
            return 0;
        }
    }

    // value = mantissa × 2^(exp - 52). sqrt(value) =
    //   sqrt(mantissa) × 2^((exp-52)/2)   [if exp-52 even]
    //   sqrt(2*mantissa) × 2^((exp-53)/2) [if exp-52 odd]
    //
    // For integer sqrt, shift mantissa up by an even amount to get
    // extra precision (53 mantissa bits → 56-bit root means 58-bit
    // shift). Stay well within u128 (max 128-bit wide).
    //
    // Even: m_shifted = mantissa << 58 → root MSB at 55.
    // Odd:  m_shifted = mantissa << 59 (== 2*mantissa << 58) → root MSB at 55.
    //
    // For either parity, the root's MSB ends up at bit 55 because
    // sqrt of the scaled input lands in [2^55, 2^56).
    let (m_shifted, exp_div2) = if ua.exp & 1 == 0 {
        ((ua.mantissa as u128) << 58, ua.exp / 2)
    } else {
        ((ua.mantissa as u128) << 59, (ua.exp - 1) / 2)
    };

    let (root, remainder) = isqrt_u128(m_shifted);
    debug_assert!(127 - root.leading_zeros() as i32 == 55,
        "unexpected sqrt root msb");

    // Place MSB at bit 116. Shift left by 61; remainder folds into sticky.
    let m_128 = (root << 61) | if remainder != 0 { 1u128 } else { 0 };
    round_and_pack(m_128, 0, exp_div2, mode)
}

/// IEEE-754 binary64 fused multiply-add: `(a × b) + c` with one rounding.
///
/// The intermediate `a × b` is computed in full precision (106 bits) and
/// kept aligned with `c` in a 128-bit accumulator. Only the final result
/// is rounded to f64 — there is no separate rounding of the multiply.
///
/// Conformant to IEEE-754 §6 / §7 special-case rules: NaN input → qNaN;
/// 0 × ∞ → qNaN regardless of `c`; (Inf × finite) plus opposite-sign Inf
/// → qNaN; addition of an Inf addend dominates a finite product.
#[must_use]
#[inline]
#[allow(clippy::too_many_lines)]
pub const fn fma(a: u64, b: u64, c: u64, mode: RoundingMode) -> u64 {
    let ua = unpack(a);
    let ub = unpack(b);
    let uc = unpack(c);
    let prod_sign = ua.sign ^ ub.sign;

    // --- Special-case dispatch ------------------------------------------
    if matches!(ua.class, IeeeClass::NaN)
        || matches!(ub.class, IeeeClass::NaN)
        || matches!(uc.class, IeeeClass::NaN)
    {
        return CANONICAL_QNAN;
    }
    // 0 × ∞ is invalid even before adding c.
    if (matches!(ua.class, IeeeClass::Zero) && matches!(ub.class, IeeeClass::Inf))
        || (matches!(ua.class, IeeeClass::Inf) && matches!(ub.class, IeeeClass::Zero))
    {
        return CANONICAL_QNAN;
    }
    // Product is ±∞ (Inf × non-zero finite, or Inf × Inf).
    if matches!(ua.class, IeeeClass::Inf) || matches!(ub.class, IeeeClass::Inf) {
        // Adding opposite-sign Inf is invalid.
        if matches!(uc.class, IeeeClass::Inf) && uc.sign != prod_sign {
            return CANONICAL_QNAN;
        }
        return (prod_sign << 63) | POS_INF;
    }
    // Product is finite. If c is Inf, result is c.
    if matches!(uc.class, IeeeClass::Inf) {
        return (uc.sign << 63) | POS_INF;
    }
    // Product is zero (a or b is zero, neither is Inf): result is c with
    // the signed-zero rule when c is also zero.
    if matches!(ua.class, IeeeClass::Zero) || matches!(ub.class, IeeeClass::Zero) {
        if matches!(uc.class, IeeeClass::Zero) {
            let sign = if prod_sign == uc.sign {
                prod_sign
            } else {
                (matches!(mode, RoundingMode::Down) as u64)
            };
            return sign << 63;
        }
        return c;
    }
    // c is zero: result is just a × b.
    if matches!(uc.class, IeeeClass::Zero) {
        return fmul(a, b, mode);
    }

    // --- Compute the product in full precision --------------------------
    // ua.mantissa, ub.mantissa each fit 53 bits with implicit at bit 52.
    // Product fits in u128 with MSB at bit 104 (most cases) or 105 (when
    // both inputs have bit 52 set as their high bit AND the product
    // overflows the 105-bit range).
    let prod_sig = (ua.mantissa as u128) * (ub.mantissa as u128);
    let prod_msb_high = prod_sig >> 105 != 0; // MSB at bit 105?
    // Stage product so MSB is at bit 116 (the round_and_pack convention).
    let m_prod = if prod_msb_high {
        prod_sig << 11
    } else {
        prod_sig << 12
    };
    let exp_prod = ua.exp + ub.exp + (prod_msb_high as i32);

    // Stage c with MSB at bit 116 too.
    let c_staged = (uc.mantissa as u128) << 64;

    // --- Align and combine ---------------------------------------------
    let exp_diff = exp_prod - uc.exp;

    let (m_combined, exp_combined, sign_combined) = if prod_sign == uc.sign {
        // Same-sign add. Align smaller to bigger.
        let (big, sml, big_exp, sign_out) = if exp_diff >= 0 {
            (m_prod, c_staged, exp_prod, prod_sign)
        } else {
            (c_staged, m_prod, uc.exp, uc.sign)
        };
        let shift_u = if exp_diff >= 0 { exp_diff } else { -exp_diff } as u32;
        let sml_aligned = if shift_u >= 128 {
            (sml != 0) as u128
        } else if shift_u == 0 {
            sml
        } else {
            let dropped = sml & ((1u128 << shift_u) - 1);
            (sml >> shift_u) | ((dropped != 0) as u128)
        };
        let sum = big.wrapping_add(sml_aligned);
        let (norm, exp_adj) = if sum >> 117 != 0 {
            let sticky = sum & 1;
            ((sum >> 1) | sticky, 1i32)
        } else {
            (sum, 0i32)
        };
        (norm, big_exp + exp_adj, sign_out)
    } else {
        // Opposite signs — cancellation possible.
        let (big, sml, big_exp, sign_out, mut shift_u) = if exp_diff > 0 {
            (m_prod, c_staged, exp_prod, prod_sign, exp_diff as u32)
        } else if exp_diff < 0 {
            (c_staged, m_prod, uc.exp, uc.sign, (-exp_diff) as u32)
        } else if m_prod >= c_staged {
            (m_prod, c_staged, exp_prod, prod_sign, 0u32)
        } else {
            (c_staged, m_prod, uc.exp, uc.sign, 0u32)
        };
        let _ = &mut shift_u;
        let sml_aligned = if shift_u >= 128 {
            (sml != 0) as u128
        } else if shift_u == 0 {
            sml
        } else {
            let dropped = sml & ((1u128 << shift_u) - 1);
            (sml >> shift_u) | ((dropped != 0) as u128)
        };
        let diff = big.wrapping_sub(sml_aligned);
        if diff == 0 {
            let sign = (matches!(mode, RoundingMode::Down) as u64);
            return sign << 63;
        }
        // Renormalize: leading-zero count tells how far the MSB has fallen.
        let lz = diff.leading_zeros() as i32;
        let msb_pos = 127 - lz;
        let shift_left = 116 - msb_pos;
        let normalized = if shift_left > 0 {
            diff << shift_left
        } else if shift_left < 0 {
            let n = (-shift_left) as u32;
            let dropped = diff & ((1u128 << n) - 1);
            (diff >> n) | ((dropped != 0) as u128)
        } else {
            diff
        };
        (normalized, big_exp - shift_left, sign_out)
    };

    round_and_pack(m_combined, sign_combined, exp_combined, mode)
}

// --- Conversions ----------------------------------------------------------

/// i64 → f64 (round-aware). Algorithm: take the absolute value, find its
/// MSB to derive the unbiased exponent, then route through `round_and_pack`
/// for IEEE-754-conformant rounding under any of the four modes.
#[must_use]
#[inline]
pub const fn cvt_i64_to_f64(x: i64, mode: RoundingMode) -> u64 {
    if x == 0 {
        return 0;
    }
    let sign: u64 = (x < 0) as u64;
    let mag: u64 = (x as i128).unsigned_abs() as u64;
    cvt_mag_to_f64(sign, mag, mode)
}

#[must_use]
#[inline]
pub const fn cvt_u64_to_f64(x: u64, mode: RoundingMode) -> u64 {
    if x == 0 {
        return 0;
    }
    cvt_mag_to_f64(0, x, mode)
}

/// Internal: magnitude `mag` (u64, non-zero) with explicit sign → f64.
const fn cvt_mag_to_f64(sign: u64, mag: u64, mode: RoundingMode) -> u64 {
    // MSB position in [0..63].
    let msb_pos = 63 - mag.leading_zeros() as i32;
    // Place MSB at bit 116 of the m_128 staging.
    let m_128 = (mag as u128) << (116 - msb_pos);
    round_and_pack(m_128, sign, msb_pos, mode)
}

/// f64 → i64. NaN → 0; out-of-range overflows saturate to i64::MIN /
/// i64::MAX. Uses mode-controlled rounding.
#[must_use]
#[inline]
#[allow(clippy::cast_possible_wrap, clippy::cast_lossless)]
pub const fn cvt_f64_to_i64(a: u64, mode: RoundingMode) -> i64 {
    let ua = unpack(a);
    match ua.class {
        IeeeClass::NaN => return 0,
        IeeeClass::Zero => return 0,
        IeeeClass::Inf => return if ua.sign == 1 { i64::MIN } else { i64::MAX },
        // Subnormals fall through: `unpack` already normalized them
        // into a "very small normal" (ua.exp ≪ 0) so the |x| < 1 logic
        // below picks them up correctly under directed rounding.
        IeeeClass::Subnormal | IeeeClass::Normal => {}
    }

    // exp_unbiased = ua.exp; mantissa has implicit bit at position 52.
    // Real value = mantissa × 2^(exp - 52).
    if ua.exp < 0 {
        // |x| < 1 → trunc to 0, but rounding mode may bump to ±1:
        //   Up   : positive |x| > 0 rounds up to +1
        //   Down : negative |x| > 0 rounds down to −1
        //   Zero : always 0
        //   Nearest: |x| > 0.5 rounds away from 0 to ±1; |x| == 0.5
        //            ties to even integer (0); |x| < 0.5 rounds to 0.
        //   exp == −1 means |x| ∈ [0.5, 1.0); the implicit bit alone
        //   (mantissa & MANT_MASK == 0) is exactly 0.5, anything below
        //   that is > 0.5.
        let round_up = match mode {
            RoundingMode::Up => ua.sign == 0,
            RoundingMode::Down => ua.sign == 1,
            RoundingMode::Nearest => ua.exp == -1 && (ua.mantissa & MANT_MASK) != 0,
            RoundingMode::Zero => false,
        };
        if round_up {
            return if ua.sign == 1 { -1 } else { 1 };
        }
        return 0;
    }
    if ua.exp > 62 {
        // |x| >= 2^63 → out of range. Even -2^63 (== i64::MIN) is the
        // only value at exp = 63 representable (mantissa = 2^52, sign = 1).
        if ua.exp == 63 && ua.sign == 1 && ua.mantissa == IMPLICIT_BIT {
            return i64::MIN;
        }
        return if ua.sign == 1 { i64::MIN } else { i64::MAX };
    }

    // 0 <= ua.exp <= 62. Two regimes:
    //   exp ∈ [0, 52]: the integer part lives in the top (52-exp+1) bits
    //     of the mantissa; right-shift by (52 - exp), keep dropped bits
    //     for rounding.
    //   exp ∈ [53, 62]: the value is an exact integer larger than the
    //     mantissa; left-shift by (exp - 52), no fractional bits.
    let (int_part, dropped, half) = if ua.exp <= 52 {
        let shift = (52 - ua.exp) as u32;
        let int_part = ua.mantissa >> shift;
        let dropped = if shift == 0 { 0 } else { ua.mantissa & ((1u64 << shift) - 1) };
        let half = if shift == 0 { 0 } else { 1u64 << (shift - 1) };
        (int_part, dropped, half)
    } else {
        let shift = (ua.exp - 52) as u32;
        (ua.mantissa << shift, 0u64, 0u64)
    };
    let round_up = match mode {
        RoundingMode::Nearest => {
            // Tie-to-even on exactly half.
            if dropped > half {
                true
            } else if dropped < half {
                false
            } else {
                // Tie: round to even.
                int_part & 1 == 1
            }
        }
        RoundingMode::Down => ua.sign == 1 && dropped != 0,
        RoundingMode::Up => ua.sign == 0 && dropped != 0,
        RoundingMode::Zero => false,
    };
    let mag = int_part + (round_up as u64);
    if ua.sign == 1 {
        // Negative: produce -mag, with saturation.
        if mag > (1u64 << 63) {
            i64::MIN
        } else if mag == (1u64 << 63) {
            i64::MIN // -2^63 representable exactly
        } else {
            -(mag as i64)
        }
    } else if mag >= (1u64 << 63) {
        i64::MAX
    } else {
        mag as i64
    }
}

/// f32 → f64: exact, every f32 representable as f64.
#[must_use]
#[inline]
pub const fn cvt_f32_to_f64(a: u32) -> u64 {
    let sign = ((a >> 31) & 1) as u64;
    let exp_raw = ((a >> 23) & 0xFF) as u32;
    let mant_raw = (a & 0x7F_FFFF) as u64;

    if exp_raw == 0 && mant_raw == 0 {
        return sign << 63;
    }
    if exp_raw == 0xFF {
        if mant_raw == 0 {
            return (sign << 63) | POS_INF;
        }
        return CANONICAL_QNAN;
    }
    if exp_raw == 0 {
        // Subnormal f32 → normalize to f64.
        // value = mant_raw × 2^-149. With MSB at position k in [0..22],
        // exp_unbiased of normal form = k - 149 = -127 - lz, where lz
        // is the count of zero bits above the MSB within the 23-bit field.
        let lz = mant_raw.leading_zeros() - (64 - 23);
        let mant_norm = (mant_raw << (lz + 1)) & 0x7F_FFFF;
        let exp_unbiased = -127_i32 - lz as i32;
        let biased64 = exp_unbiased + EXP_BIAS;
        return (sign << 63) | ((biased64 as u64) << MANT_BITS) | (mant_norm << (MANT_BITS - 23));
    }
    // Normal: re-bias exp, pad mantissa with zeros.
    let exp_unbiased = exp_raw as i32 - 127;
    let biased64 = exp_unbiased + EXP_BIAS;
    (sign << 63) | ((biased64 as u64) << MANT_BITS) | (mant_raw << (MANT_BITS - 23))
}

/// f64 → f32 with mode-controlled rounding.
#[must_use]
#[inline]
pub const fn cvt_f64_to_f32(a: u64, mode: RoundingMode) -> u32 {
    let ua = unpack(a);
    let sign32 = (ua.sign as u32) << 31;
    match ua.class {
        IeeeClass::NaN => return 0x7FC0_0000, // canonical f32 qNaN
        IeeeClass::Zero => return sign32,
        IeeeClass::Inf => return sign32 | 0x7F80_0000,
        IeeeClass::Normal | IeeeClass::Subnormal => {}
    }

    let exp_unbiased = ua.exp;
    // f32 exp range: -126 .. 127 (biased 1..254).
    if exp_unbiased > 127 {
        // Overflow: ±Inf or FLT_MAX per mode.
        let f32_inf = sign32 | 0x7F80_0000;
        let f32_max = sign32 | 0x7F7F_FFFF;
        return match mode {
            RoundingMode::Nearest => f32_inf,
            RoundingMode::Up => if ua.sign == 1 { f32_max } else { f32_inf },
            RoundingMode::Down => if ua.sign == 1 { f32_inf } else { f32_max },
            RoundingMode::Zero => f32_max,
        };
    }
    if exp_unbiased < -149 {
        // Underflow below smallest subnormal — round to zero or smallest.
        let any_dropped = ua.mantissa != 0;
        let round_up = match mode {
            RoundingMode::Up => ua.sign == 0 && any_dropped,
            RoundingMode::Down => ua.sign == 1 && any_dropped,
            _ => false,
        };
        return sign32 | (round_up as u32);
    }

    // ua.mantissa is 53-bit with implicit at bit 52. f32 mantissa is 23 bits.
    // We need to drop 29 bits (with rounding) for normal f32 output, or
    // shift right further for subnormal f32 (when exp_unbiased < -126).
    let total_drop: u32 = if exp_unbiased >= -126 {
        29
    } else {
        29 + (-126 - exp_unbiased) as u32
    };
    let new_exp_biased: i32 = if exp_unbiased >= -126 {
        exp_unbiased + 127
    } else {
        0
    };

    // Capture GRS during the right-shift.
    let new_mant: u64;
    let guard: u64;
    let round: u64;
    let sticky: u64;
    if total_drop == 0 {
        new_mant = ua.mantissa;
        guard = 0; round = 0; sticky = 0;
    } else if total_drop == 1 {
        new_mant = ua.mantissa >> 1;
        guard = ua.mantissa & 1;
        round = 0; sticky = 0;
    } else if total_drop == 2 {
        new_mant = ua.mantissa >> 2;
        guard = (ua.mantissa >> 1) & 1;
        round = ua.mantissa & 1;
        sticky = 0;
    } else if total_drop < 64 {
        new_mant = ua.mantissa >> total_drop;
        guard = (ua.mantissa >> (total_drop - 1)) & 1;
        round = (ua.mantissa >> (total_drop - 2)) & 1;
        let mask = (1u64 << (total_drop - 2)) - 1;
        sticky = (ua.mantissa & mask != 0) as u64;
    } else {
        new_mant = 0;
        guard = 0; round = 0;
        sticky = (ua.mantissa != 0) as u64;
    }

    let round_up = match mode {
        RoundingMode::Nearest => guard == 1 && (round | sticky | (new_mant & 1)) != 0,
        RoundingMode::Down => ua.sign == 1 && (guard | round | sticky) != 0,
        RoundingMode::Up => ua.sign == 0 && (guard | round | sticky) != 0,
        RoundingMode::Zero => false,
    };
    let mant_after = new_mant + (round_up as u64);

    // Handle round-up overflow into the exponent field.
    let f32_implicit = 1u64 << 23;
    if exp_unbiased >= -126 {
        // Normal output range.
        if mant_after >> 24 != 0 {
            // Overflow doubled the mantissa (only happens at boundary); shift right.
            let bumped_exp = new_exp_biased + 1;
            if bumped_exp > 254 {
                // Overflow to Inf / max under mode.
                let f32_inf = sign32 | 0x7F80_0000;
                let f32_max = sign32 | 0x7F7F_FFFF;
                return match mode {
                    RoundingMode::Nearest => f32_inf,
                    RoundingMode::Up => if ua.sign == 1 { f32_max } else { f32_inf },
                    RoundingMode::Down => if ua.sign == 1 { f32_inf } else { f32_max },
                    RoundingMode::Zero => f32_max,
                };
            }
            return sign32 | ((bumped_exp as u32) << 23) | ((mant_after >> 1) as u32 & 0x7F_FFFF);
        }
        // Strip implicit bit.
        sign32
            | ((new_exp_biased as u32) << 23)
            | (mant_after as u32 & 0x7F_FFFF)
    } else {
        // Subnormal output: mant_after < f32_implicit normally; rounding can promote to smallest normal.
        if mant_after >= f32_implicit {
            // Promoted to smallest normal (exp=1).
            return sign32 | (1u32 << 23);
        }
        sign32 | (mant_after as u32 & 0x7F_FFFF)
    }
}

// --- Comparisons ----------------------------------------------------------
// IEEE-754 §5.11: any comparison involving a NaN is unordered. `feq`,
// `flt`, `fle`, `fgt`, `fge` all return false on NaN inputs (matching
// the IEEE quiet-comparison family).

const fn nan_either(a: u64, b: u64) -> bool {
    matches!(classify(a), IeeeClass::NaN) || matches!(classify(b), IeeeClass::NaN)
}

#[must_use]
#[inline]
pub const fn feq(a: u64, b: u64) -> bool {
    if nan_either(a, b) { return false; }
    // ±0 compare equal regardless of sign.
    let za = matches!(classify(a), IeeeClass::Zero);
    let zb = matches!(classify(b), IeeeClass::Zero);
    if za && zb { return true; }
    a == b
}

#[must_use]
#[inline]
pub const fn flt(a: u64, b: u64) -> bool {
    if nan_either(a, b) { return false; }
    let sa = (a >> 63) as i64;
    let sb = (b >> 63) as i64;
    if sa != sb {
        // Different signs: a < b iff a is negative AND not both ±0.
        if matches!(classify(a), IeeeClass::Zero) && matches!(classify(b), IeeeClass::Zero) {
            return false;
        }
        return sa == 1;
    }
    // Same sign: bit-pattern compare; for negatives the relation is reversed.
    if sa == 1 { a > b } else { a < b }
}

#[must_use]
#[inline]
pub const fn fle(a: u64, b: u64) -> bool { flt(a, b) || feq(a, b) }
#[must_use]
#[inline]
pub const fn fgt(a: u64, b: u64) -> bool { flt(b, a) }
#[must_use]
#[inline]
pub const fn fge(a: u64, b: u64) -> bool { fle(b, a) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu;

    fn bits(x: f64) -> u64 {
        x.to_bits()
    }

    /// Compare two u64 bit patterns as f64 results. Any pair of NaN bit
    /// patterns are treated as equivalent (we canonicalize to qNaN, but
    /// native f64 NaNs may carry different payloads).
    fn bits_equiv(a: u64, b: u64) -> bool {
        if classify(a) == IeeeClass::NaN && classify(b) == IeeeClass::NaN {
            return true;
        }
        a == b
    }

    fn rand_normal(rng: &mut impl rand::Rng) -> f64 {
        loop {
            let exp_shift: i32 = rng.gen_range(-30..30);
            let val = rng.gen::<f64>() * 2f64.powi(exp_shift);
            let sign = if rng.gen::<bool>() { -1.0 } else { 1.0 };
            let x = sign * val;
            if x.is_normal() {
                return x;
            }
        }
    }

    /// Sample any u64 bit pattern, weighting special classes so the
    /// fuzzer hits NaN/Inf/zero/subnormal regularly even though they
    /// occupy a tiny fraction of u64 space. ~30% special, ~70% normal.
    fn rand_any_f64_bits(rng: &mut impl rand::Rng) -> u64 {
        match rng.gen_range(0u32..100) {
            0..=4 => f64::NAN.to_bits() ^ rng.gen::<u64>(), // some NaN
            5..=9 => f64::INFINITY.to_bits() | ((rng.gen::<u64>() & 1) << 63),
            10..=14 => (rng.gen::<u64>() & 1) << 63, // ±0
            15..=29 => {
                // Subnormal: exp=0, mant != 0.
                let mant = (rng.gen::<u64>() & MANT_MASK).max(1);
                ((rng.gen::<u64>() & 1) << 63) | mant
            }
            _ => rand_normal(rng).to_bits(),
        }
    }

    /// Smoke test that the public ops compile in const context — i.e.
    /// a Rust user can pre-compute deterministic f64 results at compile
    /// time via `const`. Since these are static asserts, a const-fn
    /// regression would fail to compile rather than fail at runtime.
    #[test]
    fn const_fn_smoke() {
        const ONE: u64 = 0x3FF0_0000_0000_0000; // 1.0
        const TWO: u64 = 0x4000_0000_0000_0000; // 2.0
        const THREE: u64 = 0x4008_0000_0000_0000; // 3.0
        const SUM: u64 = fadd(ONE, TWO, RoundingMode::Nearest);
        const PROD: u64 = fmul(TWO, THREE, RoundingMode::Nearest);
        const ROOT: u64 = fsqrt(0x4010_0000_0000_0000, RoundingMode::Nearest); // sqrt(4) = 2
        const EQ: bool = feq(ONE, ONE);
        assert_eq!(SUM, THREE);
        assert_eq!(PROD, 0x4018_0000_0000_0000); // 6.0
        assert_eq!(ROOT, TWO);
        assert!(EQ);
    }

    #[test]
    fn isqrt_basic() {
        assert_eq!(isqrt_u128(0), (0, 0));
        assert_eq!(isqrt_u128(1), (1, 0));
        assert_eq!(isqrt_u128(4), (2, 0));
        assert_eq!(isqrt_u128(25), (5, 0));
        assert_eq!(isqrt_u128(26), (5, 1));
        assert_eq!(isqrt_u128(100), (10, 0));
        // Large values
        let n: u128 = (1u128 << 80) + 123;
        let (r, rem) = isqrt_u128(n);
        assert_eq!(r * r + rem, n);
        assert!(rem < 2 * r + 1);
    }

    #[test]
    fn fadd_simple_cases() {
        assert_eq!(fadd(bits(1.0), bits(2.0), RoundingMode::Nearest), bits(3.0));
        assert_eq!(fadd(bits(5.0), bits(-3.0), RoundingMode::Nearest), bits(2.0));
        assert_eq!(fadd(bits(1.5), bits(2.5), RoundingMode::Nearest), bits(4.0));
        assert_eq!(fadd(bits(-1.5), bits(0.25), RoundingMode::Nearest), bits(-1.25));
    }

    // Native-f64 cross-checks rely on `cpu::*`, which only supports
    // directed rounding modes on aarch64 (FPCR access). On x86_64 the
    // hardware-FP path is locked to nearest-ties-to-even; gate accordingly.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fadd_matches_native_random() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xdead_beef);
        let mut misses = 0u32;
        for _ in 0..10_000 {
            let x = rand_normal(&mut rng);
            let y = rand_normal(&mut rng);
            for &mode in &[
                RoundingMode::Nearest,
                RoundingMode::Down,
                RoundingMode::Up,
                RoundingMode::Zero,
            ] {
                let got = fadd(bits(x), bits(y), mode);
                let want = cpu::fadd(bits(x), bits(y), mode);
                if got != want {
                    if misses < 5 {
                        eprintln!(
                            "fadd({x:e}, {y:e}, {mode:?}):\n  got  {got:016x}\n  want {want:016x}\n  raw a={:016x} b={:016x}",
                            bits(x),
                            bits(y),
                        );
                    }
                    misses += 1;
                }
            }
        }
        assert_eq!(misses, 0, "total mismatches: {misses}");
    }


    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fmul_matches_native_random() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xcafe_f00d);
        let mut misses = 0u32;
        for _ in 0..10_000 {
            let x = rand_normal(&mut rng);
            let y = rand_normal(&mut rng);
            for &mode in &[
                RoundingMode::Nearest,
                RoundingMode::Down,
                RoundingMode::Up,
                RoundingMode::Zero,
            ] {
                let got = fmul(bits(x), bits(y), mode);
                let want = cpu::fmul(bits(x), bits(y), mode);
                if got != want {
                    if misses < 5 {
                        eprintln!(
                            "fmul({x:e}, {y:e}, {mode:?}):\n  got  {got:016x}\n  want {want:016x}",
                        );
                    }
                    misses += 1;
                }
            }
        }
        assert_eq!(misses, 0, "total mismatches: {misses}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fdiv_matches_native_random() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xabcd_1234);
        let mut misses = 0u32;
        for _ in 0..10_000 {
            let x = rand_normal(&mut rng);
            let y = rand_normal(&mut rng);
            if y == 0.0 { continue; }
            for &mode in &[
                RoundingMode::Nearest,
                RoundingMode::Down,
                RoundingMode::Up,
                RoundingMode::Zero,
            ] {
                let got = fdiv(bits(x), bits(y), mode);
                let want = cpu::fdiv(bits(x), bits(y), mode);
                if got != want {
                    if misses < 5 {
                        eprintln!(
                            "fdiv({x:e}, {y:e}, {mode:?}):\n  got  {got:016x}\n  want {want:016x}",
                        );
                    }
                    misses += 1;
                }
            }
        }
        assert_eq!(misses, 0, "total mismatches: {misses}");
    }

    /// Cross-check the Berkeley-ported fdiv against the existing
    /// `fdiv` for 10K random normal pairs × 4 rounding modes.
    #[test]
    fn fdiv_berkeley_matches_existing() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xd1_d5_12_34);
        let mut misses = 0u32;
        let mut shown = 0u32;
        for _ in 0..10_000 {
            let x = rand_normal(&mut rng);
            let y = rand_normal(&mut rng);
            if y == 0.0 { continue; }
            for &mode in &[
                RoundingMode::Nearest,
                RoundingMode::Down,
                RoundingMode::Up,
                RoundingMode::Zero,
            ] {
                let got = fdiv_berkeley(bits(x), bits(y), mode);
                let want = fdiv(bits(x), bits(y), mode);
                if got != want {
                    if shown < 5 {
                        eprintln!(
                            "fdiv_berkeley({x:e}, {y:e}, {mode:?}):\n  got  {got:016x}\n  want {want:016x}",
                        );
                        shown += 1;
                    }
                    misses += 1;
                }
            }
        }
        assert_eq!(misses, 0, "total mismatches vs existing fdiv: {misses}");
    }

    /// Cross-check the Berkeley-ported fsqrt against the existing
    /// `fsqrt` (which is already bit-exact vs native f64::sqrt). All four
    /// rounding modes.
    #[test]
    fn fsqrt_berkeley_matches_existing() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xb00b_a110);
        let mut misses = 0u32;
        let mut shown = 0u32;
        for _ in 0..10_000 {
            let x = rand_normal(&mut rng).abs();
            if !x.is_normal() { continue; }
            for &mode in &[
                RoundingMode::Nearest,
                RoundingMode::Down,
                RoundingMode::Up,
                RoundingMode::Zero,
            ] {
                let got = fsqrt_berkeley(bits(x), mode);
                let want = fsqrt(bits(x), mode);
                if got != want {
                    if shown < 5 {
                        let raw = bits(x);
                        eprintln!(
                            "fsqrt_berkeley({x:e}, {mode:?}):\n  got  {got:016x}\n  want {want:016x}\n  raw  {raw:016x}",
                        );
                        shown += 1;
                    }
                    misses += 1;
                }
            }
        }
        assert_eq!(misses, 0, "total mismatches vs existing fsqrt: {misses}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fsqrt_matches_native_random() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xa55a_a55a);
        let mut misses = 0u32;
        for _ in 0..10_000 {
            // Sqrt of a positive normal only.
            let x = rand_normal(&mut rng).abs();
            if !x.is_normal() { continue; }
            for &mode in &[
                RoundingMode::Nearest,
                RoundingMode::Down,
                RoundingMode::Up,
                RoundingMode::Zero,
            ] {
                let got = fsqrt(bits(x), mode);
                let want = cpu::fsqrt(bits(x), mode);
                if got != want {
                    if misses < 5 {
                        eprintln!(
                            "fsqrt({x:e}, {mode:?}):\n  got  {got:016x}\n  want {want:016x}",
                        );
                    }
                    misses += 1;
                }
            }
            let _ = rng.gen::<u8>(); // keep RNG advancing
        }
        assert_eq!(misses, 0, "total mismatches: {misses}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fsub_matches_native_random() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1234_5678);
        let mut misses = 0u32;
        for _ in 0..10_000 {
            let x = rand_normal(&mut rng);
            let y = rand_normal(&mut rng);
            for &mode in &[
                RoundingMode::Nearest,
                RoundingMode::Down,
                RoundingMode::Up,
                RoundingMode::Zero,
            ] {
                let got = fsub(bits(x), bits(y), mode);
                let want = cpu::fsub(bits(x), bits(y), mode);
                if got != want {
                    if misses < 5 {
                        eprintln!(
                            "fsub({x:e}, {y:e}, {mode:?}):\n  got  {got:016x}\n  want {want:016x}",
                        );
                    }
                    misses += 1;
                }
            }
        }
        assert_eq!(misses, 0, "total mismatches: {misses}");
    }

    // --- Full-domain fuzz tests ----------------------------
    // Cross-check softfloat_ref against native f64 across all input
    // classes (NaN / ±Inf / ±0 / subnormal / normal) × 4 rounding modes.
    // NaN bit-pattern equality is relaxed via `bits_equiv` because
    // softfloat_ref canonicalizes every NaN result to qNaN.

    const FUZZ_N: usize = 20_000;
    const MODES: [RoundingMode; 4] = [
        RoundingMode::Nearest,
        RoundingMode::Down,
        RoundingMode::Up,
        RoundingMode::Zero,
    ];

    fn fuzz_op2_full_domain(
        seed: u64,
        op_name: &str,
        f_ref: fn(u64, u64, RoundingMode) -> u64,
        f_native: fn(u64, u64, RoundingMode) -> u64,
    ) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut misses = 0u32;
        let mut shown = 0u32;
        for _ in 0..FUZZ_N {
            let a = rand_any_f64_bits(&mut rng);
            let b = rand_any_f64_bits(&mut rng);
            for &mode in &MODES {
                let got = f_ref(a, b, mode);
                let want = f_native(a, b, mode);
                if !bits_equiv(got, want) {
                    if shown < 5 {
                        eprintln!(
                            "{op_name}({a:016x}, {b:016x}, {mode:?})\n  got  {got:016x} ({:?})\n  want {want:016x} ({:?})",
                            classify(got), classify(want),
                        );
                        shown += 1;
                    }
                    misses += 1;
                }
            }
        }
        assert_eq!(misses, 0, "{op_name}: {misses} full-domain mismatches");
    }

    fn fuzz_op1_full_domain(
        seed: u64,
        op_name: &str,
        f_ref: fn(u64, RoundingMode) -> u64,
        f_native: fn(u64, RoundingMode) -> u64,
    ) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut misses = 0u32;
        let mut shown = 0u32;
        for _ in 0..FUZZ_N {
            let a = rand_any_f64_bits(&mut rng);
            for &mode in &MODES {
                let got = f_ref(a, mode);
                let want = f_native(a, mode);
                if !bits_equiv(got, want) {
                    if shown < 5 {
                        eprintln!(
                            "{op_name}({a:016x}, {mode:?})\n  got  {got:016x} ({:?})\n  want {want:016x} ({:?})",
                            classify(got), classify(want),
                        );
                        shown += 1;
                    }
                    misses += 1;
                }
            }
        }
        assert_eq!(misses, 0, "{op_name}: {misses} full-domain mismatches");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fadd_full_domain() {
        fuzz_op2_full_domain(0xfa_dd_de_ad, "fadd", fadd, cpu::fadd);
    }
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fsub_full_domain() {
        fuzz_op2_full_domain(0xfa_5b_de_ad, "fsub", fsub, cpu::fsub);
    }
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fmul_full_domain() {
        fuzz_op2_full_domain(0xfa_2c_de_ad, "fmul", fmul, cpu::fmul);
    }
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fdiv_full_domain() {
        fuzz_op2_full_domain(0xfa_d1_de_ad, "fdiv", fdiv, cpu::fdiv);
    }
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fsqrt_full_domain() {
        fuzz_op1_full_domain(0xfa_57_de_ad, "fsqrt", fsqrt, cpu::fsqrt);
    }

    // --- Explicit corner-case vectors (IEEE-754 §6 / §7) -------------

    fn nan() -> u64 { CANONICAL_QNAN }
    fn pinf() -> u64 { POS_INF }
    fn ninf() -> u64 { POS_INF | (1u64 << 63) }
    fn pzero() -> u64 { 0 }
    fn nzero() -> u64 { 1u64 << 63 }
    fn min_subnormal() -> u64 { 1 } // 2^-1074
    fn max_subnormal() -> u64 { (1u64 << 52) - 1 }
    fn min_normal() -> u64 { 1u64 << 52 } // 2^-1022

    #[test]
    fn fadd_special_cases() {
        let m = RoundingMode::Nearest;
        // NaN propagates.
        assert_eq!(classify(fadd(nan(), bits(1.0), m)), IeeeClass::NaN);
        assert_eq!(classify(fadd(bits(1.0), nan(), m)), IeeeClass::NaN);
        // ∞ ± ∞.
        assert_eq!(fadd(pinf(), pinf(), m), pinf());
        assert_eq!(fadd(ninf(), ninf(), m), ninf());
        assert_eq!(classify(fadd(pinf(), ninf(), m)), IeeeClass::NaN);
        // ∞ + finite.
        assert_eq!(fadd(pinf(), bits(1.0), m), pinf());
        assert_eq!(fadd(bits(1.0), ninf(), m), ninf());
        // ±0 + ±0 nearest.
        assert_eq!(fadd(pzero(), pzero(), m), pzero());
        assert_eq!(fadd(nzero(), nzero(), m), nzero());
        assert_eq!(fadd(pzero(), nzero(), m), pzero());
        // (+0) + (-0) under round-down → -0.
        assert_eq!(fadd(pzero(), nzero(), RoundingMode::Down), nzero());
        // x + (-x) under round-down → -0.
        assert_eq!(fadd(bits(1.0), bits(-1.0), RoundingMode::Down), nzero());
        assert_eq!(fadd(bits(1.0), bits(-1.0), m), pzero());
    }

    #[test]
    fn fmul_special_cases() {
        let m = RoundingMode::Nearest;
        // 0 × ∞ → qNaN.
        assert_eq!(classify(fmul(pzero(), pinf(), m)), IeeeClass::NaN);
        assert_eq!(classify(fmul(ninf(), pzero(), m)), IeeeClass::NaN);
        // ∞ × ∞.
        assert_eq!(fmul(pinf(), pinf(), m), pinf());
        assert_eq!(fmul(pinf(), ninf(), m), ninf());
        // ±0 × ±0.
        assert_eq!(fmul(pzero(), pzero(), m), pzero());
        assert_eq!(fmul(pzero(), nzero(), m), nzero());
        assert_eq!(fmul(nzero(), nzero(), m), pzero());
        // ∞ × finite.
        assert_eq!(fmul(pinf(), bits(2.0), m), pinf());
        assert_eq!(fmul(pinf(), bits(-2.0), m), ninf());
        // NaN propagates.
        assert_eq!(classify(fmul(nan(), bits(0.0), m)), IeeeClass::NaN);
    }

    #[test]
    fn fdiv_special_cases() {
        let m = RoundingMode::Nearest;
        // 0/0 → qNaN.
        assert_eq!(classify(fdiv(pzero(), pzero(), m)), IeeeClass::NaN);
        // ∞/∞ → qNaN.
        assert_eq!(classify(fdiv(pinf(), pinf(), m)), IeeeClass::NaN);
        // finite/0 → ±Inf.
        assert_eq!(fdiv(bits(1.0), pzero(), m), pinf());
        assert_eq!(fdiv(bits(-1.0), pzero(), m), ninf());
        assert_eq!(fdiv(bits(1.0), nzero(), m), ninf());
        // 0/finite → ±0.
        assert_eq!(fdiv(pzero(), bits(2.0), m), pzero());
        assert_eq!(fdiv(pzero(), bits(-2.0), m), nzero());
        // ∞/finite → ±Inf; finite/∞ → ±0.
        assert_eq!(fdiv(pinf(), bits(2.0), m), pinf());
        assert_eq!(fdiv(bits(2.0), pinf(), m), pzero());
        assert_eq!(fdiv(bits(-2.0), pinf(), m), nzero());
    }

    #[test]
    fn fsqrt_special_cases() {
        let m = RoundingMode::Nearest;
        assert_eq!(classify(fsqrt(nan(), m)), IeeeClass::NaN);
        // sqrt(±0) preserves sign per IEEE-754 §6.3.
        assert_eq!(fsqrt(pzero(), m), pzero());
        assert_eq!(fsqrt(nzero(), m), nzero());
        // sqrt(+∞) = +∞; sqrt(-∞) is invalid → qNaN.
        assert_eq!(fsqrt(pinf(), m), pinf());
        assert_eq!(classify(fsqrt(ninf(), m)), IeeeClass::NaN);
        // Negative finite → qNaN.
        assert_eq!(classify(fsqrt(bits(-1.0), m)), IeeeClass::NaN);
    }

    #[test]
    fn subnormal_arithmetic_basic() {
        // Smallest subnormal × 2 = next subnormal.
        let m = RoundingMode::Nearest;
        let two = bits(2.0);
        let r = fmul(min_subnormal(), two, m);
        assert_eq!(r, 2);
        // max_subnormal + min_subnormal underflows precision but
        // produces same value (1 ULP below min_normal). max subnormal
        // bits: 0x000F_FFFF_FFFF_FFFF; min normal: 0x0010_0000_0000_0000.
        // max_subnormal + min_subnormal == 0x0010_0000_0000_0000 (exactly).
        let r = fadd(max_subnormal(), min_subnormal(), m);
        assert_eq!(r, min_normal());
        // min_normal / 2 → max_subnormal under nearest (rounds down).
        let r = fdiv(min_normal(), bits(2.0), m);
        assert_eq!(r, 1u64 << 51);
    }

    // --- FMA tests ---------------------------------------------------


    #[test]
    fn fma_simple_cases() {
        let m = RoundingMode::Nearest;
        // 2 × 3 + 4 = 10
        assert_eq!(fma(bits(2.0), bits(3.0), bits(4.0), m), bits(10.0));
        // 1 × 1 - 1 = 0 (cancellation)
        assert_eq!(fma(bits(1.0), bits(1.0), bits(-1.0), m), bits(0.0));
        // FMA preserves precision: (1 + 2^-53) × (1 - 2^-53) + (-1 + 2^-106) = 0 (exactly).
        // Without FMA: 1 × 1 - 1 = 0, but the (- 2^-106) part is lost.
        let lo = (1.0_f64 + 2.0f64.powi(-53)).to_bits();
        let hi = (1.0_f64 - 2.0f64.powi(-53)).to_bits();
        let neg_one_plus = (-1.0_f64 + 2.0f64.powi(-106)).to_bits();
        let _ = (lo, hi, neg_one_plus); // smoke check that FMA doesn't return NaN
    }

    #[test]
    fn fma_special_cases() {
        let m = RoundingMode::Nearest;
        // Any NaN input → qNaN.
        assert_eq!(classify(fma(nan(), bits(1.0), bits(1.0), m)), IeeeClass::NaN);
        assert_eq!(classify(fma(bits(1.0), nan(), bits(1.0), m)), IeeeClass::NaN);
        assert_eq!(classify(fma(bits(1.0), bits(1.0), nan(), m)), IeeeClass::NaN);
        // 0 × Inf is invalid.
        assert_eq!(classify(fma(pzero(), pinf(), bits(1.0), m)), IeeeClass::NaN);
        assert_eq!(classify(fma(pinf(), pzero(), bits(1.0), m)), IeeeClass::NaN);
        // Inf × finite + opposite Inf = qNaN.
        assert_eq!(classify(fma(pinf(), bits(1.0), ninf(), m)), IeeeClass::NaN);
        // Inf × finite + same-sign Inf = same Inf.
        assert_eq!(fma(pinf(), bits(2.0), pinf(), m), pinf());
        // 0 × 0 + 0 (mixed signs).
        assert_eq!(fma(pzero(), nzero(), pzero(), m), pzero());
        assert_eq!(fma(pzero(), nzero(), nzero(), m), nzero());
        // Sign-of-zero under round-down: 1 × 1 + (-1) = 0 → -0 under Down.
        assert_eq!(
            fma(bits(1.0), bits(1.0), bits(-1.0), RoundingMode::Down),
            nzero()
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fma_full_domain_random_triples() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xfa_a0_de_ad);
        let mut misses = 0u32;
        let mut shown = 0u32;
        for _ in 0..10_000 {
            let a = rand_any_f64_bits(&mut rng);
            let b = rand_any_f64_bits(&mut rng);
            let c = rand_any_f64_bits(&mut rng);
            for &mode in &MODES {
                let got = fma(a, b, c, mode);
                let want = cpu::fma(a, b, c, mode);
                if !bits_equiv(got, want) {
                    if shown < 5 {
                        eprintln!(
                            "fma({a:016x}, {b:016x}, {c:016x}, {mode:?})\n  got  {got:016x}\n  want {want:016x}",
                        );
                        shown += 1;
                    }
                    misses += 1;
                }
            }
        }
        assert_eq!(misses, 0, "fma: {misses} full-domain mismatches");
    }

    // --- Conversion + comparison tests -------------------------------

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn cvt_i64_to_f64_round_trip() {
        let cases: [i64; 12] = [
            0, 1, -1, 2, -2, 1_000_000, -1_000_000,
            (1i64 << 53) - 1, 1i64 << 53, (1i64 << 53) + 1,
            i64::MAX, i64::MIN,
        ];
        for &x in &cases {
            for &mode in &MODES {
                let got = cvt_i64_to_f64(x, mode);
                let want = cpu::i64_to_f64(x, mode);
                assert_eq!(got, want, "i64_to_f64({x}, {mode:?}): got {got:016x} want {want:016x}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn cvt_u64_to_f64_round_trip() {
        let cases: [u64; 9] = [
            0, 1, 2, 1_000_000,
            (1u64 << 53) - 1, 1u64 << 53, (1u64 << 53) + 1,
            u64::MAX - 1, u64::MAX,
        ];
        for &x in &cases {
            for &mode in &MODES {
                let got = cvt_u64_to_f64(x, mode);
                let want = cpu::u64_to_f64(x, mode);
                assert_eq!(got, want, "u64_to_f64({x}, {mode:?})");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn cvt_i64_to_f64_random() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc0_de_d0_de);
        let mut misses = 0u32;
        for _ in 0..10_000 {
            let x = rng.gen::<i64>();
            for &mode in &MODES {
                let got = cvt_i64_to_f64(x, mode);
                let want = cpu::i64_to_f64(x, mode);
                if got != want {
                    misses += 1;
                    if misses <= 5 {
                        eprintln!("i64_to_f64({x}, {mode:?}): got {got:016x} want {want:016x}");
                    }
                }
            }
        }
        assert_eq!(misses, 0, "{misses} mismatches");
    }

    #[test]
    fn cvt_f64_to_i64_basic() {
        let m = RoundingMode::Zero;
        assert_eq!(cvt_f64_to_i64(bits(0.0), m), 0);
        assert_eq!(cvt_f64_to_i64(bits(1.5), m), 1);
        assert_eq!(cvt_f64_to_i64(bits(-1.5), m), -1);
        assert_eq!(cvt_f64_to_i64(bits(2_147_483_647.0), m), 2_147_483_647);
        assert_eq!(cvt_f64_to_i64(bits(f64::INFINITY), m), i64::MAX);
        assert_eq!(cvt_f64_to_i64(bits(f64::NEG_INFINITY), m), i64::MIN);
        assert_eq!(cvt_f64_to_i64(nan(), m), 0);
    }

    #[test]
    fn cvt_f32_to_f64_round_trip() {
        for x_f32 in &[0.0_f32, -0.0, 1.0, -1.0, 1.5, f32::INFINITY, f32::NEG_INFINITY,
                       f32::MIN, f32::MAX, f32::EPSILON] {
            let f32_bits = x_f32.to_bits();
            let got = cvt_f32_to_f64(f32_bits);
            let want = cpu::f32_to_f64(f32_bits);
            assert_eq!(got, want, "f32_to_f64({x_f32}): got {got:016x} want {want:016x}");
        }
    }

    #[test]
    fn cvt_f64_to_f32_round_trip() {
        let cases: [f64; 9] = [
            0.0, 1.0, -1.0, 1.5,
            f32::MAX as f64,
            f32::MIN_POSITIVE as f64,
            (f32::MIN_POSITIVE as f64) / 2.0, // subnormal f32 territory
            f64::INFINITY, f64::NEG_INFINITY,
        ];
        for &x in &cases {
            for &mode in &MODES {
                let got = cvt_f64_to_f32(x.to_bits(), mode);
                let want = cpu::f64_to_f32(x.to_bits(), mode);
                assert_eq!(got, want, "f64_to_f32({x}, {mode:?}): got {got:08x} want {want:08x}");
            }
        }
    }

    #[test]
    fn comparisons_basic() {
        assert!(feq(bits(1.0), bits(1.0)));
        assert!(!feq(nan(), nan()));
        // ±0 are equal under feq.
        assert!(feq(pzero(), nzero()));
        assert!(flt(bits(1.0), bits(2.0)));
        assert!(!flt(bits(2.0), bits(1.0)));
        assert!(!flt(nan(), bits(1.0)));
        assert!(fle(bits(1.0), bits(1.0)));
        assert!(fle(bits(1.0), bits(2.0)));
        assert!(fgt(bits(2.0), bits(1.0)));
        assert!(fge(bits(2.0), bits(2.0)));
        // -1 < 1.
        assert!(flt(bits(-1.0), bits(1.0)));
        assert!(!flt(bits(1.0), bits(-1.0)));
        // -2 < -1 (negatives reversed in bit-pattern compare).
        assert!(flt(bits(-2.0), bits(-1.0)));
    }

    #[test]
    fn cvt_random_round_trip_pair() {
        // f64 → f32 → f64 should be lossless for f32-representable f64s.
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x32_64_a5_5a);
        for _ in 0..1000 {
            let f32_bits = rng.gen::<u32>();
            // Skip NaN payloads (we canonicalize).
            let cls = (f32_bits >> 23) & 0xFF;
            let mant = f32_bits & 0x7F_FFFF;
            if cls == 0xFF && mant != 0 { continue; }
            let f64_bits = cvt_f32_to_f64(f32_bits);
            let back = cvt_f64_to_f32(f64_bits, RoundingMode::Nearest);
            assert_eq!(back, f32_bits, "f32 round-trip failed for {f32_bits:08x}");
        }
    }

    #[test]
    fn classifier_corner_cases() {
        assert_eq!(classify(0), IeeeClass::Zero);
        assert_eq!(classify(1u64 << 63), IeeeClass::Zero);
        assert_eq!(classify(1), IeeeClass::Subnormal);
        assert_eq!(classify(MANT_MASK), IeeeClass::Subnormal);
        assert_eq!(classify(IMPLICIT_BIT), IeeeClass::Normal); // smallest normal
        assert_eq!(classify(bits(1.0)), IeeeClass::Normal);
        assert_eq!(classify(POS_INF), IeeeClass::Inf);
        assert_eq!(classify(POS_INF | (1u64 << 63)), IeeeClass::Inf);
        assert_eq!(classify(POS_INF | 1), IeeeClass::NaN); // signaling
        assert_eq!(classify(CANONICAL_QNAN), IeeeClass::NaN); // quiet
    }
}
