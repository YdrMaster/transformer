use std::{borrow::Borrow, collections::HashMap, hash::Hash, ops::Deref};

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
