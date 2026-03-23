use super::Cpu;
use core::arch::wasm32::*;

pub struct CurrentCpu {}

const STEP: usize = 16;
const EPR: usize = 4;
const ARR: usize = STEP / EPR;

impl Cpu<ARR> for CurrentCpu {
    type Unit = v128;
    type Array = [v128; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;

    fn n() -> usize {
        ARR
    }

    unsafe fn zero() -> Self::Unit {
        f32x4_splat(0.0)
    }

    unsafe fn zero_array() -> Self::Array {
        [Self::zero(); ARR]
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        f32x4_splat(v)
    }

    unsafe fn load(mem_addr: *const f32) -> Self::Unit {
        v128_load(mem_addr as *mut v128)
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        f32x4_add(a, b)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        f32x4_add(f32x4_mul(b, c), a)
    }

    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit) {
        v128_store(mem_addr as *mut v128, a);
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        for i in 0..ARR / 2 {
            x[2 * i] = f32x4_add(x[2 * i], x[2 * i + 1]);
        }
        for i in 0..ARR / 4 {
            x[4 * i] = f32x4_add(x[4 * i], x[4 * i + 2]);
        }
        for i in 0..ARR / 8 {
            x[8 * i] = f32x4_add(x[8 * i], x[8 * i + 4]);
        }
        *y = f32x4_extract_lane::<0>(x[0])
            + f32x4_extract_lane::<1>(x[0])
            + f32x4_extract_lane::<2>(x[0])
            + f32x4_extract_lane::<3>(x[0]);
    }
}

/// Compute 4 dot products simultaneously: dot(a, b0), dot(a, b1), dot(a, b2), dot(a, b3).
/// Shares LHS loads across all 4 RHS columns for better throughput.
/// All pointers must point to contiguous f32 arrays of length >= k.
#[inline(always)]
pub(crate) unsafe fn vec_dot_f32_4col(
    a: *const f32,
    b0: *const f32,
    b1: *const f32,
    b2: *const f32,
    b3: *const f32,
    k: usize,
) -> [f32; 4] {
    let np = k & !(STEP - 1);

    // 4 independent accumulators per column to break dependency chains
    let mut sum0 = [f32x4_splat(0.0); ARR];
    let mut sum1 = [f32x4_splat(0.0); ARR];
    let mut sum2 = [f32x4_splat(0.0); ARR];
    let mut sum3 = [f32x4_splat(0.0); ARR];

    for i in (0..np).step_by(STEP) {
        for j in 0..ARR {
            let av = v128_load(a.add(i + j * EPR) as *const v128);
            sum0[j] = f32x4_add(sum0[j], f32x4_mul(av, v128_load(b0.add(i + j * EPR) as *const v128)));
            sum1[j] = f32x4_add(sum1[j], f32x4_mul(av, v128_load(b1.add(i + j * EPR) as *const v128)));
            sum2[j] = f32x4_add(sum2[j], f32x4_mul(av, v128_load(b2.add(i + j * EPR) as *const v128)));
            sum3[j] = f32x4_add(sum3[j], f32x4_mul(av, v128_load(b3.add(i + j * EPR) as *const v128)));
        }
    }

    // Tree-reduce each column's accumulators
    #[inline(always)]
    unsafe fn reduce(mut s: [v128; ARR]) -> f32 {
        for i in 0..ARR / 2 {
            s[2 * i] = f32x4_add(s[2 * i], s[2 * i + 1]);
        }
        for i in 0..ARR / 4 {
            s[4 * i] = f32x4_add(s[4 * i], s[4 * i + 2]);
        }
        f32x4_extract_lane::<0>(s[0])
            + f32x4_extract_lane::<1>(s[0])
            + f32x4_extract_lane::<2>(s[0])
            + f32x4_extract_lane::<3>(s[0])
    }

    let mut result = [reduce(sum0), reduce(sum1), reduce(sum2), reduce(sum3)];

    // Scalar remainder for k % 16
    for i in np..k {
        let av = *a.add(i);
        result[0] += av * *b0.add(i);
        result[1] += av * *b1.add(i);
        result[2] += av * *b2.add(i);
        result[3] += av * *b3.add(i);
    }

    result
}
