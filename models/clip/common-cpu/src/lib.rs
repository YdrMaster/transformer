use clip::{ClipStorage, WeightLoader};
use operators::{add_rows, common_cpu::Cpu, conv, QueueOf, TopoNode};
use std::marker::PhantomData;

pub struct Operators<N = Cpu>(PhantomData<N>);

pub struct Weights<'w> {
    patch_embd_w: &'w [u8],
    patch_embd_b: &'w [u8],
    pos_embd: &'w [u8],
}

impl<N> clip::Operators for Operators<N>
where
    N: TopoNode<Cpu>,
{
    type Hardware = Cpu;
    type TopoNode = Cpu;
    type Conv = conv::common_cpu::ConvIm2Col;
    type AddRows = add_rows::common_cpu::Operator;
}

impl<'w> Weights<'w> {
    pub fn new(model: &'w ClipStorage<&'w [u8]>) -> Self {
        Self {
            patch_embd_w: model.patch_embd_w,
            patch_embd_b: model.patch_embd_b,
            pos_embd: model.pos_embd,
        }
    }
}

impl WeightLoader for Weights<'_> {
    type Hardware = Cpu;
    type Weight<'s> = &'s [u8] where Self: 's;

    #[inline]
    fn patch_embd<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> [Self::Weight<'a>; 2] {
        [self.patch_embd_w, self.patch_embd_b]
    }

    #[inline]
    fn pos_embd<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a> {
        self.pos_embd
    }
}

#[cfg(test)]
mod test_infer;
