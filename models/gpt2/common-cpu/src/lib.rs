
use gpt2::{
    ext::ggml_quants::{self, digit_layout::DigitLayout, f16, DataBlock, QuantExt},
    storage::{BlkStorage, Storage},
    BlkWeight, GPT2BlkWeight, Tensor, WeightLoader,
};
pub use gpt2::{GPT2BlkStorage, GPT2Storage,TensorUsage::Computation};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    common_cpu::Cpu,
    random_sample::common_cpu::Operator as RandomSampleCpu,
    rearrange::common_cpu::Operator as Rearrange,
    Blob, ByteOf, QueueOf, TopoNode,
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
    pos_embd: DigitLayout,
    dt_embd: DigitLayout,
    dt_mat: DigitLayout,
    nexp: usize,
    size_qkv: usize,
    size_o: usize,
    size_gate_up: usize,
    size_down: usize,
}

pub struct WeightCache {
    cache: Blob,
    cached_weight: GPT2BlkWeight,
    cached_weight_iblk: usize,
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
    type AddRows = op!(add_rows);
    type LayerNorm = op!(layer_norm);
    type MatMul = op!(mat_mul);
    type AttnKVCached = op!(attention_kv_cached);
    type Gelu = op!(gelu);
    type Add = op!(add);
    type Rearrange = op!(rearrange);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        println!("{tensor}");
    }
}

impl<'w> Weights<'w> {
    pub fn new(model: &'w GPT2Storage<&'w [u8]>, dist: Distribution) -> Self {
        let GPT2Storage {
            meta,
            output,
            blocks,
            token_embd,
            pos_embd,
            output_norm_b,
            output_norm_w,
        } = model;

        let blks = blocks
            .iter()
            .map(|blk| {
                blk.clone()
                    .into_vec()
                    .into_iter()
                    .map(|(which, data)| {
                        (which, meta.distribute_data(which, data, dist, Blob::new))
                    })
                    .collect::<GPT2BlkStorage<_>>()
            })
            .collect::<Box<_>>();

        let meta = meta.distribute(dist);
        let size_qkv_w = meta.attn_qkv_w(Computation).take();
        let size_qkv_b = meta.attn_qkv_b(Computation).take();
        let size_o_w = meta.attn_o_w(Computation).take();
        let size_o_b = meta.attn_o_b(Computation).take();
        let size_up_w = meta.ffn_down_w(Computation).take();
        let size_up_b = meta.ffn_down_b(Computation).take();
        let size_down_w = meta.ffn_down_w(Computation).take();
        let size_down_b = meta.ffn_down_b(Computation).take();
        Self {
            blks,
            output_norm_b,
            output_norm_w,
            output,
            dt_embd: meta.dt_embd,
            dt_mat: meta.dt_linear,
            nexp: meta.nexp,
            size_qkv,
            size_o,
            size_gate_up,
            size_down,
            pos_embd,
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
