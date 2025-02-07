#![cfg(detected)]

use common::Distribution;
use llama::{LlamaBlkStorage, LlamaBlkWeight, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    clrt::{Context, SvmBlob, SvmByte},
    opencl::ClDevice,
    random_sample::opencl::Operator as RandomSampleCl,
    rearrange::opencl::Operator as Rearrange,
    Blob, ByteOf, QueueOf, TopoNode,
};
use std::{marker::PhantomData, ops::Deref, ptr::copy_nonoverlapping};

pub struct Operators<N = ClDevice, R = NonAllReduce<ClDevice, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<ClDevice, RandomSampleCl>;

macro_rules! op {
    ($name:ident) => {
        operators::$name::opencl::Operator
    };
}

impl<N, R> llama::Operators for Operators<N, R>
where
    N: TopoNode<ClDevice>,
    R: AllReduce<ClDevice, N>,
{
    type Hardware = ClDevice;
    type TopoNode = N;
    type RmsNorm = op!(rms_norm);
    type MatMul = op!(mat_mul);
    type Rope = op!(rope);
    type AttnKVCached = op!(attention_kv_cached);
    type Swiglu = op!(swiglu);
    type Rearrange = op!(rearrange);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>, queue: &QueueOf<Self::Hardware>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        let tensor = tensor.as_ref().map(|s| queue.map(s));
        println!("{tensor}");
        queue.unmap(tensor.take())
    }

    fn memcpy_d2h<T: Copy>(
        dst: &mut [T],
        src: &[ByteOf<Self::Hardware>],
        queue: &QueueOf<Self::Hardware>,
    ) {
        assert_eq!(size_of_val(dst), size_of_val(src));
        let svm = queue.map(src);
        unsafe { copy_nonoverlapping(svm.as_ptr(), dst.as_mut_ptr().cast::<u8>(), dst.len()) }
        queue.unmap(svm)
    }
}

pub struct Weights {
    nexp: usize,
    blks: Box<[LlamaBlkStorage<SvmBlob>]>,
    output_norm: SvmBlob,
    output: SvmBlob,
}

impl Weights {
    pub fn new(model: &LlamaStorage<&[u8]>, dist: Distribution, ctx: &Context) -> Self {
        let LlamaStorage {
            meta,
            output_norm,
            output,
            blocks,
            ..
        } = model;

        let meta = meta.distribute(dist);
        let queue = ctx.queue();
        let blks = blocks
            .iter()
            .map(|blk| {
                blk.clone()
                    .into_vec()
                    .into_iter()
                    .map(|(which, data)| {
                        let blob = meta.distribute_data(which, data, dist, Blob::new);
                        let mut svm = ctx.malloc::<u8>(blob.len());
                        let mut map = queue.map_mut(&mut svm, false);
                        map.copy_from_slice(&blob);
                        queue.unmap(map);
                        (which, svm)
                    })
                    .collect::<LlamaBlkStorage<_>>()
            })
            .collect::<Vec<_>>();

        let mut output_norm_svm = ctx.malloc::<u8>(output_norm.len());
        let mut output_svm = ctx.malloc::<u8>(output.len());
        let mut output_norm_map = queue.map_mut(&mut output_norm_svm, false);
        let mut output_map = queue.map_mut(&mut output_svm, false);
        output_norm_map.copy_from_slice(output_norm);
        output_map.copy_from_slice(output);
        queue.unmap(output_norm_map);
        queue.unmap(output_map);

        Self {
            nexp: meta.nexp,
            blks: blks.into_boxed_slice(),
            output_norm: output_norm_svm,
            output: output_svm,
        }
    }
}

impl WeightLoader for Weights {
    type Hardware = ClDevice;
    type Weight<'s>
        = &'s [SvmByte]
    where
        Self: 's;

    #[inline]
    fn load_blk(
        &self,
        which: LlamaBlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Weight<'_> {
        let LlamaBlkStorage {
            attn_norm,
            attn_qkv,
            attn_qkv_bias,
            attn_o,
            ffn_norm,
            ffn_gate_inp,
            ffn_gate_up,
            ffn_down,
        } = &self.blks[iblk];

        use LlamaBlkWeight as W;
        #[rustfmt::skip]
        let ans = match which {
            W::AttnNorm    => attn_norm    ,
            W::AttnQKV     => attn_qkv     ,
            W::AttnQKVBias => attn_qkv_bias,
            W::AttnO       => attn_o       ,
            W::FfnNorm     => ffn_norm     ,
            W::FfnGateInp  => ffn_gate_inp ,
            W::FfnGateUp   => ffn_gate_up  ,
            W::FfnDown     => ffn_down     ,
        };
        ans
    }

    fn load_moe<'a>(
        &'a self,
        which: LlamaBlkWeight,
        iblk: usize,
        iexp: usize,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a> {
        let LlamaBlkStorage {
            ffn_gate_up,
            ffn_down,
            ..
        } = &self.blks[iblk];

        let w = match which {
            LlamaBlkWeight::FfnGateUp => ffn_gate_up,
            LlamaBlkWeight::FfnDown => ffn_down,
            _ => unreachable!(),
        };
        let one = w.len() / self.nexp;
        &w[iexp * one..][..one]
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        &self.output_norm
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        &self.output
    }
}

#[cfg(test)]
mod infer;
