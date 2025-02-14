use std::{
    ops::{Deref, DerefMut},
    rc::Rc,
    slice::from_raw_parts_mut,
};

pub trait Splitable {
    fn split(&self) -> Self;
}

impl<T> Splitable for &[T] {
    #[inline]
    fn split(&self) -> Self {
        self
    }
}

impl<T> Splitable for &mut [T] {
    #[inline]
    fn split(&self) -> Self {
        unsafe { from_raw_parts_mut(self.as_ptr().cast_mut(), self.len()) }
    }
}

#[repr(transparent)]
pub struct LocalSplitable<T>(Rc<T>);

impl<T> From<T> for LocalSplitable<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self(Rc::new(t))
    }
}

impl<T> Splitable for LocalSplitable<T> {
    #[inline]
    fn split(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Deref> Deref for LocalSplitable<T> {
    type Target = T::Target;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<T, U> DerefMut for LocalSplitable<T>
where
    T: DerefMut<Target = [U]>,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let data = self.0.as_ptr().cast_mut();
        let len = self.0.len();
        unsafe { from_raw_parts_mut(data, len) }
    }
}

#[macro_export]
macro_rules! split {
    ($tensor:expr => $( $name:ident ),+; [$( $part:expr ),+] @ $axis:expr) => {
        let parts = [$($part),+];
        let mut parts = $tensor.split($axis, &parts);
        $( let $name = parts.next().unwrap(); )+
        assert!(parts.next().is_none());
    };
}
#[macro_export]
macro_rules! split_mut {
    ($tensor:expr => $( $name:ident ),+; [$( $part:expr ),+] @ $axis:expr) => {
        let parts = [$($part),+];
        let mut parts = $tensor.split($axis, &parts);
        $( let mut $name = parts.next().unwrap(); )+
        assert!(parts.next().is_none());
    };
}
