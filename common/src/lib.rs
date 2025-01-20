use std::{
    borrow::Borrow,
    collections::HashMap,
    hash::Hash,
    ops::{Deref, Range},
};

pub enum Contiguous<'a, T> {
    Borrowed(&'a [u8]),
    Owned(T),
}

impl<T: Deref<Target = [u8]>> Deref for Contiguous<'_, T> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(b) => b,
            Self::Owned(o) => o,
        }
    }
}

#[inline(always)]
pub fn borrow<T>(t: &[u8]) -> Contiguous<'_, T> {
    Contiguous::Borrowed(t)
}

#[inline(always)]
pub fn own<'a, T>(t: T) -> Contiguous<'a, T> {
    Contiguous::Owned(t)
}

#[derive(Clone, Default, Debug)]
#[repr(transparent)]
pub struct Slab<K, V>(HashMap<K, Vec<V>>);

impl<K, V> Slab<K, V> {
    #[inline]
    pub fn new() -> Self {
        Self(HashMap::new())
    }
}

impl<K: Eq + Hash, V> Slab<K, V> {
    #[inline]
    pub fn take<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.0.get_mut(key).and_then(|pool| pool.pop())
    }

    #[inline]
    pub fn put(&mut self, key: K, value: V) {
        self.0.entry(key).or_default().push(value);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Distribution {
    start: usize,
    len: usize,
    total: usize,
}

impl Distribution {
    pub const MONO: Self = Self {
        start: 0,
        len: 1,
        total: 1,
    };

    pub fn new(start: usize, len: usize, total: usize) -> Self {
        assert!(0 < len && start + len <= total);
        Self { start, len, total }
    }

    #[inline]
    pub const fn is_mono(&self) -> bool {
        self.len == self.total
    }

    #[inline]
    pub const fn info(&self) -> [usize; 3] {
        [self.start, self.len, self.total]
    }
}

pub struct WeightMemCalculator {
    align: usize,
    size: usize,
}

impl WeightMemCalculator {
    #[inline]
    pub const fn new(align: usize) -> Self {
        Self { align, size: 0 }
    }

    #[inline]
    pub const fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn push(&mut self, size: usize) -> Range<usize> {
        let start = self.size.div_ceil(self.align) * self.align;
        self.size = start + size;
        start..self.size
    }
}
