use std::arch::x86_64::*;

pub(super) const AVX2_U16_W: usize = 16;

pub(super) const fn round_u16(x: u32) -> u16 {
    // 座標の最大値は2^22 = 4_194_304とする（さすがに大丈夫やろ……）
    // これを16bitに収めるためには、6bit右シフトすればよい（64単位で丸められる）
    // 事前に1 << 5を足しておくと四捨五入になる
    ((x + (1 << 5)) >> 6) as u16
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

impl Into<__m256i> for AlignedU16 {
    fn into(self) -> __m256i {
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
