use gpt2::{
    storage::{BlkStorage, Storage},
    BlkWeight, Tensor, WeightLoader,
};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    common_cpu::Cpu,
    random_sample::common_cpu::Operator as RandomSampleCpu,
    rearrange::common_cpu::Operator as Rearrange,
    ByteOf, QueueOf, TopoNode,
};
use std::marker::PhantomData;
use std::ops::Deref;

pub struct Operators<N = Cpu, R = NonAllReduce<Cpu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = gpt2::RandomSample<Cpu, RandomSampleCpu>;

pub struct Weights<'w> {
    blks: Box<[BlkStorage<&'w [u8]>]>,
    output_norm_w: &'w [u8],
    output_norm_b: &'w [u8],
    output: &'w [u8],
    pos_embd: &'w [u8],
}

macro_rules! op {
    ($name:ident) => {
        operators::$name::common_cpu::Operator
    };
}

impl<N, R> gpt2::Operators for Operators<N, R>
where
    N: TopoNode<Cpu>,
    R: AllReduce<Cpu, N>,
{
    type Hardware = Cpu;
    type TopoNode = N;
    type LayerNorm = op!(layer_norm);
    type MatMul = op!(mat_mul);
    type AttnKVCached = op!(attention_kv_cached);
    type Rearrange = op!(rearrange);
    type AllReduce = R;
    type AddRows = op!(add_rows);
    type Mlp = op!(gpt2_mlp);
    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        println!("{tensor}");
    }
}

impl<'w> Weights<'w> {
    pub fn new(model: &'w Storage<&'w [u8]>) -> Self {
        let Storage {
            output_norm_w,
            output_norm_b,
            output,
            blocks,
            pos_embd,
            ..
        } = model;

        Self {
            pos_embd,
            blks: blocks.clone(),
            output_norm_w,
            output_norm_b,
            output,
        }
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
        let BlkStorage {
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
        } = &self.blks[iblk];

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
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> [Self::Memory<'_>; 2] {
        [self.output_norm_w, self.output_norm_b]
    }
    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        self.output
    }
    #[inline]
    fn pos_embd<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a> {
        self.pos_embd
    }
}

#[cfg(test)]
pub mod infer;
