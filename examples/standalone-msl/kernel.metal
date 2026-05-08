#include <metal_stdlib>
using namespace metal;

#include "softfloat64.metal"

struct In  { ulong a; ulong b; };

kernel void compute_fadd(
    device const In* inputs [[buffer(0)]],
    device ulong*    out    [[buffer(1)]],
    constant uint&   mode   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = __softfloat64_fadd(inputs[gid].a, inputs[gid].b, mode);
}
