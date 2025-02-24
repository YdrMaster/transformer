use common::{Contiguous, Distribution};
use minicpm3::{MiniCPM3BlkStorage, MiniCPM3BlkWeight, MiniCPM3Storage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    common_cpu::Cpu,
    random_sample::common_cpu::Operator as RandomSampleCpu,
    rearrange::common_cpu::Operator as Rearrange,
    Blob, ByteOf, QueueOf, TopoNode,
};
use std::{marker::PhantomData, ops::Deref};

pub struct Operators<N = Cpu, R = NonAllReduce<Cpu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = minicpm3::RandomSample<Cpu, RandomSampleCpu>;

pub struct Weights<'w> {
    blks: Box<[MiniCPM3BlkStorage<Contiguous<'w, Blob>>]>,
    output_norm: &'w [u8],
    output: &'w [u8],
    long_factor: &'w [u8],
    sort_factor: &'w [u8],
}

macro_rules! op {
    ($name:ident) => {
        operators::$name::common_cpu::Operator
    };
}

impl<N, R> minicpm3::Operators for Operators<N, R>
where
    N: TopoNode<Cpu>,
    R: AllReduce<Cpu, N>,
{
    type Hardware = Cpu;
    type TopoNode = N;
    type Rope = op!(rope);
    type AttentionMLA = op!(attention_mla);
    type RmsNorm = op!(rms_norm);
    type Add = op!(add);
    type MatMul = op!(mat_mul);
    type Swiglu = op!(swiglu);
    type Rearrange = op!(rearrange);
    type Scale = op!(scale);
    type AttentionMLACached =  op!(attention_mla_cached);
    type FuesdSoftmax = op!(fuesd_softmax);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>, _queue: &QueueOf<Self::Hardware>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        println!("{tensor}")
    }
}

impl<'w> Weights<'w> {
    pub fn new(model: &'w MiniCPM3Storage<&'w [u8]>, dist: Distribution) -> Self {
        let MiniCPM3Storage {
            meta,
            output_norm,
            output,
            blocks,
            rope_long,
            rope_short,
            ..
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
                    .collect::<MiniCPM3BlkStorage<_>>()
            })
            .collect();

        Self {
            blks,
            output_norm,
            output,
            long_factor: rope_long,
            sort_factor: rope_short,
        }
    }
}

impl WeightLoader for Weights<'_> {
    type Hardware = Cpu;
    type Weight<'s>
        = &'s [u8]
    where
        Self: 's;

    #[inline]
    fn load_blk(
        &self,
        which: MiniCPM3BlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Weight<'_> {
        let MiniCPM3BlkStorage {
            attn_norm,
            attn_qb,
            attn_qa,
            attn_kvb,
            attn_kva,
            attn_qa_norm,
            attn_kva_norm,
            attn_o,
            ffn_norm,
            ffn_gate_up,
            ffn_down,
            ffn_gate,
            ffn_up,
        } = &self.blks[iblk];
        use MiniCPM3BlkWeight as W;
        match which {
            W::AttnNorm => attn_norm,
            W::AttnQB => attn_qb,
            W::AttnQA => attn_qa,
            W::AttnKvB => attn_kvb,
            W::AttnKvA => attn_kva,
            W::AttnQANorm => attn_qa_norm,
            W::AttnKvANorm => attn_kva_norm,
            W::AttnO => attn_o,
            W::FfnNorm => ffn_norm,
            W::FfnGateUp => ffn_gate_up,
            W::FfnDown => ffn_down,
            W::FfnGate => ffn_gate,
            W::FfnUp => ffn_up,
        }
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        self.output_norm
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        self.output
    }
    #[inline]
    fn long_factor<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        self.long_factor
    }
    #[inline]
    fn short_factor<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        self.sort_factor
    }
}

#[cfg(test)]
mod infer;
