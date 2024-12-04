use clip::{ClipStorage, WeightLoader};
use operators::{common_cpu::Cpu, conv, QueueOf, TopoNode};
use std::marker::PhantomData;

pub struct Operators<N = Cpu>(PhantomData<N>);

#[repr(transparent)]
pub struct Weights<'w>(ClipStorage<&'w [u8]>);

macro_rules! op {
    ($name:ident) => {
        operators::$name::common_cpu::Operator
    };
}

impl<N> clip::Operators for Operators<N>
where
    N: TopoNode<Cpu>,
{
    type Hardware = Cpu;
    type TopoNode = Cpu;
    type Conv = conv::common_cpu::ConvIm2Col;
    type AddRows = op!(add_rows);
    type LayerNorm = op!(layer_norm);
}

impl<'w> Weights<'w> {
    pub fn new(model: &'w ClipStorage<&'w [u8]>) -> Self {
        Self(model.clone())
    }
}

impl WeightLoader for Weights<'_> {
    type Hardware = Cpu;
    type Weight<'s>
        = &'s [u8]
    where
        Self: 's;

    #[inline]
    fn patch_embd<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> [Self::Weight<'a>; 2] {
        [self.0.patch_embd_w, self.0.patch_embd_b]
    }

    #[inline]
    fn pos_embd<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a> {
        self.0.pos_embd
    }

    #[inline]
    fn pre_norm<'a>(
        &'a self,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Option<[Self::Weight<'a>; 2]> {
        self.0.pre_norm
    }

    #[inline]
    fn post_norm<'a>(
        &'a self,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Option<[Self::Weight<'a>; 2]> {
        self.0.post_norm
    }
}

#[cfg(test)]
mod test_infer;
