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
use BlkWeight::{
    AttnNormb, AttnNormw, AttnOb, AttnOw, AttnQKVb, AttnQKVw, FfnDownb, FfnDownw, FfnNormb,
    FfnNormw, FfnUpb, FfnUpw,
};

pub struct Operators<N = Cpu, R = NonAllReduce<Cpu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = gpt2::RandomSample<Cpu, RandomSampleCpu>;

pub struct Weights<'w> {
    blks: Box<[BlkStorage<&'w [u8]>]>,
    output_norm_weight: &'w [u8],
    output_norm_bias: &'w [u8],
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
            output_norm_weight,
            output_norm_bias,
            output,
            blocks,
            pos_embd,
            ..
        } = model;

        Self {
            pos_embd,
            blks: blocks.clone(),
            output_norm_weight,
            output_norm_bias,
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

    #[inline]
    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Memory<'_> {
        let BlkStorage {
            attn_norm_weight,
            attn_norm_bias,
            attn_qkv_weight,
            attn_qkv_bias,
            attn_output_weight,
            attn_output_bias,

            ffn_norm_weight,
            ffn_norm_bias,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
        } = &self.blks[iblk];

        match which {
            AttnNormw => attn_norm_weight,
            AttnNormb => attn_norm_bias,
            AttnQKVw => attn_qkv_weight,
            AttnQKVb => attn_qkv_bias,
            AttnOw => attn_output_weight,
            AttnOb => attn_output_bias,

            FfnNormw => ffn_norm_weight,
            FfnNormb => ffn_norm_bias,
            FfnUpw => ffn_up_weight,
            FfnUpb => ffn_up_bias,
            FfnDownw => ffn_down_weight,
            FfnDownb => ffn_down_bias,
        }
    }

    #[inline]
    fn output_norm_weight(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        self.output_norm_weight
    }

    #[inline]
    fn output_norm_bias(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        self.output_norm_bias
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
