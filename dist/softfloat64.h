/* dist/softfloat64.h — host-side companion header for softfloat64.metal.
 *
 * SPDX-License-Identifier: BSD-3-Clause AND MIT
 *
 * The MSL kernel API (`__softfloat64_*` in dist/softfloat64.metal) is a
 * shader-side surface. C, Objective-C, C++, and Swift hosts dispatch
 * those kernels via Metal; they do not link against the functions
 * directly. This header exists so hosts can:
 *
 *   1. Use named constants for the IEEE-754 rounding-mode argument
 *      instead of magic 0/1/2/3 ints.
 *   2. Mirror the layout of `__softfloat64_unp` when building buffer
 *      structs that carry unpacked-state values across the host/GPU
 *      boundary (used only by the optional unpack/pack adapter API —
 *      the bit-pattern API takes plain `uint64_t` and needs nothing
 *      from this header).
 *
 * The simple bit-pattern API (fadd/fsub/fmul/fdiv/fsqrt/fma/cmp/
 * conversions) just shuttles `uint64_t` (f64), `int64_t` (i64), and
 * `uint32_t` (f32). Swift: `Double(bitPattern:)` / `.bitPattern`. C:
 * `memcpy` between `double` and `uint64_t`.
 */

#ifndef SOFTFLOAT64_H
#define SOFTFLOAT64_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* IEEE-754 rounding modes. Pass as the `mode` argument to any
 * __softfloat64_* kernel function. */
enum {
    SOFTFLOAT64_ROUND_NEAREST_TIES_TO_EVEN = 0,
    SOFTFLOAT64_ROUND_TOWARD_NEGATIVE      = 1,
    SOFTFLOAT64_ROUND_TOWARD_POSITIVE      = 2,
    SOFTFLOAT64_ROUND_TOWARD_ZERO          = 3
};

/* Mirror of MSL `struct __softfloat64_unp` (see dist/softfloat64.metal).
 * Fields, types, and order match the MSL declaration so this struct can
 * be placed directly in a `device`-bound buffer. Use only with the
 * optional `__softfloat64_unpack` / `__softfloat64_pack` /
 * `__softfloat64_*_unp_normal` adapter kernels. Treat as opaque. */
typedef struct {
    uint64_t sign;
    int32_t  exp;
    uint64_t mantissa;
} softfloat64_unp;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SOFTFLOAT64_H */
