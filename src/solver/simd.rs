use std::arch::x86_64::*;

pub(super) const SIMD_WIDTH: usize = 16;

pub(super) const fn round_u16(x: u32) -> u16 {
    // 座標の最大値は2^22 = 4_194_304とする（さすがに大丈夫やろ……）
    // これを16bitに収めるためには、6bit右シフトすればよい（64単位で丸められる）
    // 事前に1 << 5を足しておくと四捨五入になる
    ((x + (1 << 5)) >> 6) as u16
}

pub fn horizontal_add(x: __m256i) -> i32 {
    unsafe {
        let low = _mm256_extracti128_si256(x, 0);
        let high = _mm256_extracti128_si256(x, 1);
        let x = _mm_add_epi16(low, high);
        let x = _mm_hadd_epi16(x, x);
        let x = _mm_hadd_epi16(x, x);
        let x = _mm_hadd_epi16(x, x);
        let x = _mm_extract_epi16(x, 0);
        x
    }
}

pub fn horizontal_or(x: __m256i) -> u16 {
    unsafe {
        let low = _mm256_castsi256_si128(x);
        let high = _mm256_extracti128_si256(x, 1);
        let x = _mm_or_si128(low, high);

        let x = _mm_or_si128(x, _mm_srli_si128::<2>(x));
        let x = _mm_or_si128(x, _mm_srli_si128::<4>(x));
        let x = _mm_or_si128(x, _mm_srli_si128::<8>(x));
        let x = _mm_extract_epi16(x, 0);
        x as u16
    }
}

pub fn horizontal_and(x: __m256i) -> u16 {
    unsafe {
        let low = _mm256_castsi256_si128(x);
        let high = _mm256_extracti128_si256(x, 1);
        let x = _mm_and_si128(low, high);

        let x = _mm_and_si128(x, _mm_srli_si128::<2>(x));
        let x = _mm_and_si128(x, _mm_srli_si128::<4>(x));
        let x = _mm_and_si128(x, _mm_srli_si128::<8>(x));
        let x = _mm_extract_epi16(x, 0);
        x as u16
    }
}

#[derive(Debug, Clone)]
pub(super) struct SimdRectSet {
    /// 長方形の幅を16bit x 16個packしたもの
    pub(super) heights: Vec<[u16; SIMD_WIDTH]>,

    /// 長方形の高さを16bit x 16個packしたもの
    pub(super) widths: Vec<[u16; SIMD_WIDTH]>,
}

impl SimdRectSet {
    pub(super) fn new(heights: Vec<[u16; SIMD_WIDTH]>, widths: Vec<[u16; SIMD_WIDTH]>) -> Self {
        Self { heights, widths }
    }
}

#[cfg(test)]
mod test {
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn horizontal_add_test() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let data: [u16; 16] = core::array::from_fn(|_| rng.gen());
            let x = unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) };
            let sum = horizontal_add(x) as u16;

            let mut expected = 0u16;

            for &x in data.iter() {
                expected = expected.wrapping_add(x);
            }

            assert_eq!(sum, expected);
        }
    }

    #[test]
    fn horizontal_or_test() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let result = rng.gen::<i16>();
            let x: [i16; 16] = core::array::from_fn(|_| rng.gen::<i16>() & result);
            let x = unsafe { _mm256_loadu_si256(x.as_ptr() as *const __m256i) };
            let or = horizontal_or(x);

            assert_eq!(or, result as u16);
        }
    }

    #[test]
    fn horizontal_and_test() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let result = rng.gen::<i16>();
            let x: [i16; 16] = core::array::from_fn(|_| rng.gen::<i16>() | result);
            let x = unsafe { _mm256_loadu_si256(x.as_ptr() as *const __m256i) };
            let or = horizontal_and(x);

            assert_eq!(or, result as u16);
        }
    }
}
