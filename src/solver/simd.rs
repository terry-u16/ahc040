pub(super) const SIMD_WIDTH: usize = 16;

pub(super) const fn round_u16(x: u32) -> u16 {
    // 座標の最大値は2^22 = 4_194_304とする（さすがに大丈夫やろ……）
    // これを16bitに収めるためには、6bit右シフトすればよい（64単位で丸められる）
    (x >> 6) as u16
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
