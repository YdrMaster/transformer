use crate::{idim, pattern::Pattern, udim, Affine, Shape, Tensor};
use std::{cmp::Ordering, iter::zip};

impl<Physical> Tensor<Physical> {
    pub fn slice(self, dims: &[SliceDim]) -> Self {
        let (shape, affine) = build(dims, &self.shape);
        Self {
            shape,
            pattern: Pattern(affine * self.pattern.0),
            ..self
        }
    }
}

fn build(meta: &[SliceDim], input: &[udim]) -> (Shape, Affine) {
    assert_eq!(input.len(), meta.len());
    let meta = zip(meta, input)
        .map(|(d, &len)| d.normalize(len))
        .collect::<Vec<_>>();

    let shape = meta.iter().map(|d| d.len).collect::<Shape>();
    let n = meta.len();
    let affine = Affine::from_fn(n + 1, n + 1, |r, c| {
        if r == n {
            meta.get(c).map_or(1, |d| d.start as _)
        } else if r == c {
            meta.get(c).map(|d| d.step).unwrap()
        } else {
            0
        }
    });
    (shape, affine)
}

#[test]
fn test() {
    use crate::slice;
    let (shape, affine) = build(&[slice![2;1;2], slice![0;1;4], slice![1;2;3]], &[5, 6, 7]);
    assert_eq!(shape.as_slice(), &[2, 4, 3]);
    assert_eq!(
        affine.as_slice(),
        &[
            // column major
            1, 0, 0, 2, //
            0, 1, 0, 0, //
            0, 0, 2, 1, //
            0, 0, 0, 1, //
        ]
    );
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SliceDim {
    pub start: udim,
    pub step: idim,
    pub len: udim,
}

impl SliceDim {
    #[inline]
    pub fn normalize(&self, len: udim) -> Self {
        if len == 0 {
            Self {
                start: 0,
                step: 0,
                len: 0,
            }
        } else {
            match self.step.cmp(&0) {
                Ordering::Greater => {
                    assert!(self.start < len, "{self:?}/{len}");
                    Self {
                        start: self.start,
                        step: self.step,
                        len: {
                            let step = self.step as udim;
                            (len - self.start).div_ceil(step).min(self.len)
                        },
                    }
                }
                Ordering::Equal => {
                    assert!(self.start < len, "{self:?}/{len}");
                    Self {
                        start: self.start,
                        step: self.step,
                        len: self.len,
                    }
                }
                Ordering::Less => {
                    let start = self.start.min(len - 1);
                    Self {
                        start,
                        step: self.step,
                        len: {
                            let step = self.step.unsigned_abs();
                            (start + 1).div_ceil(step).min(self.len)
                        },
                    }
                }
            }
        }
    }
}

#[macro_export]
macro_rules! slice {
    [=$idx:expr] => {
        slice![$idx; 0; 1]
    };
    [<-] => {
        slice![usize::MAX; -1; usize::MAX]
    };
    [=>] => {
        slice![0; 1; usize::MAX]
    };
    [$start:expr=>] => {
        slice![$start; 1; usize::MAX]
    };
    [=>$len:expr] => {
        slice![0; 1; $len]
    };
    [$start:expr => $end:expr] => {
        slice![$start; 1; $end - $start]
    };
    [$start:expr =>=> $len:expr] => {
        slice![$start; 1; $len]
    };
    [$start:expr => $step:expr => $len:expr] => {
        slice![$start; $step; $len]
    };
    [$start:expr; $step:expr; $len:expr] => {
        $crate::SliceDim {
            start: $start as _,
            step : $step  as _,
            len  : $len   as _,
        }
    };
}

#[test]
fn test_macro() {
    assert_eq!(
        slice![5; -3; 2],
        SliceDim {
            start: 5,
            step: -3,
            len: 2,
        }
    );
    assert_eq!(slice![=2], slice![2; 0; 1]);
    assert_eq!(slice![<-], slice![usize::MAX; -1; usize::MAX]);
    assert_eq!(slice![=>], slice![0; 1; usize::MAX]);
    assert_eq!(slice![3=>], slice![3; 1; usize::MAX]);
    assert_eq!(slice![=>5], slice![0; 1; 5]);
    assert_eq!(slice![3 => 5], slice![3; 1; 2]);
    assert_eq!(slice![3 =>=> 5], slice![3; 1; 5]);
    assert_eq!(slice![3 => 2 => 5], slice![3; 2; 5]);
}
