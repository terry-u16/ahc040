//! よく使われるユーティリティ関数をまとめたモジュール

use num::PrimInt;

/// 最小値と最大値を更新するトレイト
///
/// # Examples
///
/// ```
/// use cp_lib_rs::util::ChangeMinMax;
///
/// let mut x = 10;
/// assert!(x.change_min(3));
/// assert_eq!(x, 3);
/// ```
#[allow(dead_code)]
pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

/// 多次元配列を作成する
///
/// # Examples
///
/// ```
/// use cp_lib_rs::mat;
///
/// let a = mat![0; 4; 3];
/// assert_eq!(a, vec![vec![0; 3]; 4]);
/// ```
#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { vec![$($e),*] };
	($($e:expr,)*) => { vec![$($e),*] };
	($e:expr; $d:expr) => { vec![$e; $d] };
	($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

/// 整数の二分探索を行う
///
/// # Examples
///
/// ```
/// use cp_lib_rs::util::binary_search;
///
/// let result = binary_search(0, 10, |x| x * x <= 5);
/// assert_eq!(result, 2);
/// ```
#[allow(dead_code)]
pub fn binary_search<T: PrimInt>(ok: T, ng: T, f: impl Fn(T) -> bool) -> T {
    let mut ok = ok;
    let mut ng = ng;

    while ok.max(ng) - ok.min(ng) > T::one() {
        let mid = (ok + ng) / (T::one() << 1);

        if f(mid) {
            ok = mid;
        } else {
            ng = mid;
        }
    }

    ok
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BitSetIterU128 {
    flag: u128,
}

impl BitSetIterU128 {
    pub fn new(flag: u128) -> Self {
        Self { flag }
    }
}

impl Iterator for BitSetIterU128 {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.flag != 0 {
            let lsb = self.flag.trailing_zeros() as usize;
            self.flag ^= self.flag & -(self.flag as i128) as u128;
            return Some(lsb);
        }

        None
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn binary_search_test() {
        assert_eq!(binary_search(0, 10, |x| x * x <= 5), 2);
        assert_eq!(binary_search(10, 0, |x| x * x >= 5), 3);
    }

    #[test]
    fn bitset_iter_u128_test() {
        let mut iter = BitSetIterU128 {
            flag: 0b100001101010,
        };
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(6));
        assert_eq!(iter.next(), Some(11));
        assert_eq!(iter.next(), None);
    }
}
