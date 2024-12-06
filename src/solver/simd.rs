use std::arch::x86_64::*;

pub(super) const AVX2_U16_W: usize = 16;
pub(super) const AVX2_F32_W: usize = 8;

pub(super) const fn round_u16(x: u32) -> u16 {
    // 座標の最大値は2^22 = 4_194_304とする（さすがに大丈夫やろ……）
    // これを16bitに収めるためには、6bit右シフトすればよい（64単位で丸められる）
    // 事前に1 << 5を足しておくと四捨五入になる
    ((x + (1 << 5)) >> 6) as u16
}

pub fn horizontal_add_u16(x: __m256i) -> i32 {
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

pub fn horizontal_and_u16(x: __m256i) -> u16 {
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

pub fn horizontal_or_u16(x: __m256i) -> u16 {
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

pub fn horizontal_xor_u16(x: __m256i) -> u16 {
    unsafe {
        let low = _mm256_castsi256_si128(x);
        let high = _mm256_extracti128_si256(x, 1);
        let x = _mm_xor_si128(low, high);
        let x = _mm_xor_si128(x, _mm_srli_si128::<2>(x));
        let x = _mm_xor_si128(x, _mm_srli_si128::<4>(x));
        let x = _mm_xor_si128(x, _mm_srli_si128::<8>(x));
        let x = _mm_extract_epi32(x, 0);
        x as u16
    }
}

pub fn horizontal_add_f32(x: __m256) -> f32 {
    unsafe {
        let low = _mm256_castps256_ps128(x);
        let high = _mm256_extractf128_ps(x, 1);
        let x = _mm_add_ps(low, high);

        let x = _mm_hadd_ps(x, x);
        let x = _mm_hadd_ps(x, x);
        let x = _mm_extract_ps(x, 0);

        // f32として再解釈
        f32::from_bits(x as u32)
    }
}

pub fn bitonic_sort_u16(x: __m256i) -> __m256i {
    unsafe {
        // 上位128bitが全て1、下位128bitが全て0のマスク
        let zero = _mm_setzero_si128();
        let one = _mm_cmpeq_epi16(zero, zero);
        let zero_one_mask = _mm256_set_m128i(one, zero);

        // block 1
        // block 1-1
        let shuffled = shuffle256_1_u16(x);
        let min = _mm256_min_epu16(x, shuffled);
        let max = _mm256_max_epu16(x, shuffled);
        let x = _mm256_blend_epi16::<0b01100110>(min, max);

        // block 2
        // block 2-1
        let shuffled = shuffle256_2_u16(x);
        let (min, max) = min_max_u16(x, shuffled);
        let x = _mm256_blend_epi16::<0b00111100>(min, max);

        // block 2-2
        let shuffled = shuffle256_1_u16(x);
        let (min, max) = min_max_u16(x, shuffled);
        let x = _mm256_blend_epi16::<0b01011010>(min, max);

        // block 3
        // block 3-1
        // 上位だけxorをすることで、大小関係を逆転させる
        let x = _mm256_xor_si256(x, zero_one_mask);
        let shuffled = shuffle256_4_u16(x);
        let (min, max) = min_max_u16(x, shuffled);
        let x = _mm256_blend_epi16::<0b11110000>(min, max);

        // block 3-2
        let shuffled = shuffle256_2_u16(x);
        let (min, max) = min_max_u16(x, shuffled);
        let x = _mm256_blend_epi16::<0b11001100>(min, max);

        // block 3-3
        let shuffled = shuffle256_1_u16(x);
        let (min, max) = min_max_u16(x, shuffled);
        let x = _mm256_blend_epi16::<0b10101010>(min, max);

        // 反転させたので元に戻す
        let x = _mm256_xor_si256(x, zero_one_mask);

        // block 4
        // block 4-1
        // 比較演算 + ビットマスクが速そうだが、
        // 符号なし16bit整数の比較命令がないので2つのxmmレジスタに分割して処理
        let low = _mm256_castsi256_si128(x);
        let high = _mm256_extracti128_si256(x, 1);
        let min = _mm_min_epu16(low, high);
        let max = _mm_max_epu16(low, high);
        let x = _mm256_set_m128i(max, min);

        // block 4-2
        let shuffled = shuffle256_4_u16(x);
        let (min, max) = min_max_u16(x, shuffled);
        let x = _mm256_blend_epi16::<0b11110000>(min, max);

        // block 4-3
        let shuffled = shuffle256_2_u16(x);
        let (min, max) = min_max_u16(x, shuffled);
        let x = _mm256_blend_epi16::<0b11001100>(min, max);

        // block 4-4
        let shuffled = shuffle256_1_u16(x);
        let (min, max) = min_max_u16(x, shuffled);
        let x = _mm256_blend_epi16::<0b10101010>(min, max);
        x
    }
}

fn min_max_u16(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    unsafe {
        let min = _mm256_min_epu16(x, y);
        let max = _mm256_max_epu16(x, y);
        (min, max)
    }
}

fn shuffle256_1_u16(x: __m256i) -> __m256i {
    unsafe {
        // 16bitについてはhi/lo分けないとシャッフルできない
        let shuffled = _mm256_shufflehi_epi16::<0b10110001>(x);
        let shuffled = _mm256_shufflelo_epi16::<0b10110001>(shuffled);
        shuffled
    }
}

fn shuffle256_2_u16(x: __m256i) -> __m256i {
    unsafe {
        // 16bit整数を2つまとめて32bitと見なせば良い
        _mm256_shuffle_epi32::<0b10110001>(x)
    }
}

fn shuffle256_4_u16(x: __m256i) -> __m256i {
    unsafe {
        // 16bit整数を2つまとめて32bitと見なし、2つ飛びで交換すれば良い
        _mm256_shuffle_epi32::<0b01001110>(x)
    }
}

/// 32byte境界にアライン
#[repr(align(32))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct AlignedU16(pub [u16; AVX2_U16_W]);

impl AlignedU16 {
    pub const ZERO: Self = Self([0; AVX2_U16_W]);

    pub fn load(&self) -> __m256i {
        unsafe { _mm256_load_si256(self.0.as_ptr() as *const __m256i) }
    }

    pub fn store(&mut self, x: __m256i) {
        unsafe { _mm256_store_si256(self.0.as_mut_ptr() as *mut __m256i, x) }
    }
}

impl From<__m256i> for AlignedU16 {
    fn from(x: __m256i) -> Self {
        let mut ret = Self([0; AVX2_U16_W]);
        ret.store(x);
        ret
    }
}

/// 32byte境界にアライン
#[repr(align(32))]
#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd)]
pub(super) struct AlignedF32(pub [f32; AVX2_F32_W]);

impl AlignedF32 {
    pub const ZERO: Self = Self([0.0; AVX2_F32_W]);

    pub fn load(&self) -> __m256 {
        unsafe { _mm256_load_ps(self.0.as_ptr() as *const f32) }
    }

    pub fn store(&mut self, x: __m256) {
        unsafe { _mm256_store_ps(self.0.as_mut_ptr() as *mut f32, x) }
    }
}

impl From<__m256> for AlignedF32 {
    fn from(x: __m256) -> Self {
        let mut ret = Self([0.0; AVX2_F32_W]);
        ret.store(x);
        ret
    }
}

impl Into<__m256> for AlignedF32 {
    fn into(self) -> __m256 {
        self.load()
    }
}

#[derive(Debug, Clone)]
pub(super) struct SimdRectSet {
    /// 長方形の幅を16bit x 16個packしたもの
    pub(super) heights: Vec<[u16; AVX2_U16_W]>,

    /// 長方形の高さを16bit x 16個packしたもの
    pub(super) widths: Vec<[u16; AVX2_U16_W]>,
}

impl SimdRectSet {
    pub(super) fn new(heights: Vec<[u16; AVX2_U16_W]>, widths: Vec<[u16; AVX2_U16_W]>) -> Self {
        Self { heights, widths }
    }
}

#[cfg(test)]
mod test {
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn horizontal_xor_16_test() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let data: [u16; 16] = core::array::from_fn(|_| rng.gen());
            let x = unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) };
            let xor = horizontal_xor_u16(x);

            let mut expected = 0u16;

            for &x in data.iter() {
                expected ^= x;
            }

            assert_eq!(xor, expected);
        }
    }

    #[test]
    fn horizontal_add_16_test() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let data: [u16; 16] = core::array::from_fn(|_| rng.gen());
            let x = unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) };
            let sum = horizontal_add_u16(x) as u16;

            let mut expected = 0u16;

            for &x in data.iter() {
                expected = expected.wrapping_add(x);
            }

            assert_eq!(sum, expected);
        }
    }

    #[test]
    fn horizontal_and_16_test() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let mask: u16 = rng.gen();
            let data: [u16; 16] = core::array::from_fn(|_| rng.gen::<u16>() | mask);
            let x = unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) };
            let and = horizontal_and_u16(x);

            let mut expected = 0xffff;

            for &x in data.iter() {
                expected &= x;
            }

            assert_eq!(and, expected);
        }
    }

    #[test]
    fn horizontal_or_16_test() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let mask = rng.gen::<u16>();
            let data: [u16; 16] = core::array::from_fn(|_| rng.gen::<u16>() & mask);
            let x = unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) };
            let or = horizontal_or_u16(x);

            let mut expected = 0x0000;

            for &x in data.iter() {
                expected |= x;
            }

            assert_eq!(or, expected);
        }
    }

    #[test]
    fn horizontal_add_f32_test() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let data: [f32; 8] = core::array::from_fn(|_| rng.gen_range(0.0..1.0f32));
            let x = unsafe { _mm256_loadu_ps(data.as_ptr() as *const f32) };
            let sum = horizontal_add_f32(x);

            let mut expected = 0.0;

            for &x in data.iter() {
                expected += x;
            }

            let min = expected.min(sum);
            let max = expected.max(sum);
            let relative_error = (max - min) / max;

            assert!(relative_error < 1e-4);
        }
    }

    #[test]
    fn bitonic_sort_u16_test() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let data: [u16; 16] = core::array::from_fn(|_| rng.gen());
            let x = unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) };
            let sorted = bitonic_sort_u16(x);

            let mut expected: [u16; 16] = [0; 16];
            expected.copy_from_slice(&data);
            expected.sort_unstable();

            let mut actual: [u16; 16] = [0; 16];
            unsafe { _mm256_storeu_si256(actual.as_mut_ptr() as *mut __m256i, sorted) };

            assert_eq!(actual, expected);
        }
    }
}
