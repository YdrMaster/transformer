mod fmt;
mod operators;
mod split;

use ggus::ggml_quants::digit_layout::DigitLayout;
use std::{
    ops::{Deref, DerefMut, Range},
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub use ::operators::Blob;
pub use ndarray_layout::{ArrayLayout, Endian};
pub use operators::RandomSample;
pub use split::{LocalSplitable, Splitable};

#[derive(Clone)]
pub struct Tensor<T> {
    dt: DigitLayout,
    layout: ArrayLayout<5>,
    physical: T,
}

impl Tensor<usize> {
    pub fn new(dt: DigitLayout, shape: &[usize]) -> Self {
        let ele = dt.nbytes();
        Self {
            dt,
            layout: ArrayLayout::new_contiguous(shape, Endian::BigEndian, ele),
            physical: shape.iter().product::<usize>() * ele,
        }
    }
}

/// access
impl<T> Tensor<T> {
    /// 打开数组数据类型
    pub fn destruct_array(self) -> Self {
        use ggus::ggml_quants::digit_layout::LayoutContent::{Real, Unsigned};
        use std::iter::once;

        let Self {
            dt,
            layout,
            physical,
        } = self;

        let len = dt.group_size();
        let dt = match dt.decode() {
            Unsigned { width } if len > 1 => DigitLayout::unsigned(width as _, 1),
            Real { exponent, mantissa } if len > 1 => {
                DigitLayout::real(exponent as _, mantissa as _, 1)
            }
            _ => {
                return Self {
                    dt,
                    layout,
                    physical,
                }
            }
        };
        let shape = layout
            .shape()
            .iter()
            .cloned()
            .chain(once(len))
            .collect::<Vec<_>>();
        let strides = layout
            .strides()
            .iter()
            .cloned()
            .chain(once(dt.nbytes() as _))
            .collect::<Vec<_>>();
        let offset = layout.offset();
        Tensor {
            dt,
            layout: ArrayLayout::new(&shape, &strides, offset),
            physical,
        }
    }

    /// 返回一个转换了数据类型的张量，仅用于在 group size 相同且稠密存储的情况下转换
    pub fn cast(&self, target: DigitLayout) -> Tensor<usize> {
        assert_eq!(self.dt.group_size(), target.group_size());

        let merged = self
            .layout
            .merge_free(0, self.layout.ndim())
            .expect("dense tensor is castable");
        let &[d] = merged.shape() else { unreachable!() };
        let &[s] = merged.strides() else {
            unreachable!()
        };

        let div = self.dt.nbytes() as isize;
        let mul = target.nbytes() as isize;
        assert_eq!(div, s.abs());

        let shape = self.shape();
        let strides = self.strides();

        assert!(strides.iter().all(|s| s % div == 0));
        let strides = self
            .strides()
            .iter()
            .map(|s| s / div * mul)
            .collect::<Vec<_>>();
        Tensor {
            dt: target,
            layout: ArrayLayout::new(shape, &strides, 0),
            physical: d * mul as usize,
        }
    }

    #[inline]
    pub const fn dt(&self) -> DigitLayout {
        self.dt
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.layout.offset() as _
    }

    #[inline]
    pub fn take(self) -> T {
        self.physical
    }

    #[inline]
    pub fn get(&self) -> &T {
        &self.physical
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.physical
    }

    #[inline]
    pub fn as_ref(&self) -> Tensor<&T> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            physical: &self.physical,
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> Tensor<&mut T> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            physical: &mut self.physical,
        }
    }

    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Tensor<U> {
        Tensor {
            dt: self.dt,
            layout: self.layout,
            physical: f(self.physical),
        }
    }

    /// 判断张量是否完全连续。
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        // 任意相邻的两个维度可以合并表示张量完全连续
        (2..=self.layout.ndim()).all(|i| self.layout.merge_be(i - 2, 2).is_some())
    }

    /// 判断张量是否稠密存储。
    #[inline]
    pub fn is_dense(&self) -> bool {
        // 张量为稠密存储，当：
        self.layout
            // 所有维度可以合并成一个
            .merge_be(0, self.layout.ndim())
            // 合并后元素之间步长等于元素的长度
            .is_some_and(|layout| {
                let [s] = layout.strides() else {
                    unreachable!()
                };
                s.abs() == self.dt.nbytes() as isize
            })
    }

    #[inline]
    pub fn get_contiguous_bytes(&self) -> Option<usize> {
        let layout = self.layout.merge_be(0, self.layout.ndim())?;
        let &[size] = layout.shape() else {
            unreachable!()
        };
        Some(size * self.dt.nbytes())
    }
}

impl<T, B> Tensor<T>
where
    T: Deref<Target = [B]>,
{
    /// # Safety
    ///
    /// 这个函数将在移除生命周期约束的情况下引用原始数据，对这块存储空间进行读写的安全性由开发者保证。
    #[inline]
    pub unsafe fn map_slice_static(&self) -> Tensor<&'static [B]> {
        self.as_ref()
            .map(|x| unsafe { from_raw_parts(x.as_ptr(), x.len()) })
    }

    #[inline]
    pub fn map_slice(&self) -> Tensor<&[B]> {
        self.as_ref().map(|x| &x[..])
    }

    #[inline]
    pub fn base(&self) -> *const B {
        unsafe { self.physical.as_ptr().byte_add(self.layout.offset() as _) }
    }
}

impl<T, B> Tensor<T>
where
    T: DerefMut<Target = [B]>,
{
    /// # Safety
    ///
    /// 这个函数将在移除生命周期约束的情况下引用原始数据，对这块存储空间进行读写的安全性由开发者保证。
    #[inline]
    pub unsafe fn map_slice_static_mut(&mut self) -> Tensor<&'static mut [B]> {
        self.as_mut()
            .map(|x| unsafe { from_raw_parts_mut(x.as_mut_ptr(), x.len()) })
    }

    #[inline]
    pub fn map_slice_mut(&mut self) -> Tensor<&mut [B]> {
        self.as_mut().map(|x| &mut x[..])
    }

    #[inline]
    pub fn base_mut(&mut self) -> *mut B {
        unsafe {
            self.physical
                .as_mut_ptr()
                .byte_add(self.layout.offset() as _)
        }
    }
}

/// transform
impl<T> Tensor<T> {
    #[inline]
    pub fn transpose(self, perm: &[usize]) -> Self {
        Self {
            layout: self.layout.transpose(perm),
            ..self
        }
    }

    #[inline]
    pub fn index(self, axis: usize, index: usize) -> Self {
        Self {
            layout: self.layout.index(axis, index),
            ..self
        }
    }

    #[inline]
    pub fn slice(self, axis: usize, start: usize, step: isize, len: usize) -> Self {
        Self {
            layout: self.layout.slice(axis, start, step, len),
            ..self
        }
    }

    #[inline]
    pub fn tile(self, axis: usize, tiles: &[usize]) -> Self {
        Self {
            layout: self.layout.tile_be(axis, tiles),
            ..self
        }
    }

    #[inline]
    pub fn broadcast(self, axis: usize, times: usize) -> Self {
        Self {
            layout: self.layout.broadcast(axis, times),
            ..self
        }
    }

    #[inline]
    pub fn merge(self, range: Range<usize>) -> Option<Self> {
        self.layout
            .merge_be(range.start, range.len())
            .map(|layout| Self { layout, ..self })
    }
}

impl<T: Splitable> Tensor<T> {
    pub fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> impl Iterator<Item = Self> + 'a {
        self.layout.split(axis, parts).map(|layout| Self {
            dt: self.dt,
            layout,
            physical: self.physical.split(),
        })
    }
}

pub fn rearrange<T>(dst: &mut Tensor<T>, src: &Tensor<&[u8]>)
where
    T: DerefMut<Target = [u8]>,
{
    use ::operators::{
        common_cpu::{Cpu, ThisThread},
        rearrange::{common_cpu::Operator as Rearrange, Args},
        Operator as _,
    };

    let mut dst = dst.map_slice_mut();
    let src = src.map_slice();

    Rearrange::new(&Cpu)
        .launch(
            &Args {
                dst_layout: dst.layout(),
                dst_base: dst.base_mut(),
                src_layout: src.layout(),
                src_base: src.base(),
            },
            &mut [],
            &ThisThread,
        )
        .unwrap()
}
