use clip::{BlkWeight, ClipBlkStorage, ClipStorage, Tensor, WeightLoader};
use operators::{common_cpu::Cpu, conv, ByteOf, QueueOf, TopoNode};
use std::{marker::PhantomData, ops::Deref};

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
    type MatMul = op!(mat_mul);
    type Attention = op!(attention);
    type Gelu = op!(gelu);
    type Add = op!(add);
    type Rearrange = op!(rearrange);

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        println!("{tensor}")
    }
}

impl<'w> Weights<'w> {
    pub fn new(model: &'w ClipStorage<&'w [u8]>) -> Self {
        Self(model.clone())
    }
}

impl WeightLoader for Weights<'_> {
    type Hardware = Cpu;
    type Memory<'s>
        = &'s [u8]
    where
        Self: 's;

    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> [Self::Memory<'_>; 2] {
        let ClipBlkStorage {
            attn_norm_w,
            attn_norm_b,
            attn_qkv_w,
            attn_qkv_b,
            attn_o_w,
            attn_o_b,
            ffn_norm_w,
            ffn_norm_b,
            ffn_up_w,
            ffn_up_b,
            ffn_down_w,
            ffn_down_b,
        } = &self.0.blocks[iblk];
        match which {
            BlkWeight::AttnNorm => [attn_norm_w, attn_norm_b],
            BlkWeight::AttnQKV => [attn_qkv_w, attn_qkv_b],
            BlkWeight::AttnO => [attn_o_w, attn_o_b],
            BlkWeight::FfnNorm => [ffn_norm_w, ffn_norm_b],
            BlkWeight::FfnUp => [ffn_up_w, ffn_up_b],
            BlkWeight::FfnDown => [ffn_down_w, ffn_down_b],
        }
    }

    #[inline]
    fn patch_embd<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> [Self::Memory<'a>; 2] {
        [self.0.patch_embd_w, self.0.patch_embd_b]
    }

    #[inline]
    fn pos_embd<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a> {
        self.0.pos_embd
    }

    #[inline]
    fn pre_norm<'a>(
        &'a self,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Option<[Self::Memory<'a>; 2]> {
        self.0.pre_norm
    }

    #[inline]
    fn post_norm<'a>(
        &'a self,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Option<[Self::Memory<'a>; 2]> {
        self.0.post_norm
    }
}

#[cfg(test)]
mod infer;
