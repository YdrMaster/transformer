#![cfg(detected)]

use common::{Contiguous, Distribution, Slab, WeightMemCalculator};
use llama::{LlamaBlkStorage, LlamaBlkWeight, LlamaStorage, Tensor, WeightLoader};
use log::trace;
use operators::{
    all_reduce::{infini::Operator as InfiniAllReduce, AllReduce},
    infini::{Device, InfiniNode},
    infini_rt::{DevBlob, DevByte, HostBlob},
    random_sample::infini::Operator as RandomSampleNpu,
    ByteOf, QueueOf, TopoNode,
};
use std::{
    collections::VecDeque,
    iter::zip,
    marker::PhantomData,
    ops::{Deref, Range},
    time::Instant,
};

pub struct Operators<N = InfiniNode, R = InfiniAllReduce>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Device, RandomSampleNpu>;

macro_rules! op {
    ($name:ident) => {
        operators::$name::infini::Operator
    };
}

impl<N, R> llama::Operators for Operators<N, R>
where
    N: TopoNode<Device>,
    R: AllReduce<Device, N>,
{
    type Hardware = Device;
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
        let tensor = tensor.as_ref().map(|s| {
            let mut host = vec![0u8; s.len()];
            queue.get_device().memcpy_d2h(&mut host, s);
            queue.synchronize();
            host
        });
        println!("{tensor}")
    }

    fn memcpy_d2h<T: Copy>(
        dst: &mut [T],
        src: &[ByteOf<Self::Hardware>],
        queue: &QueueOf<Self::Hardware>,
    ) {
        queue.get_device().memcpy_d2h(dst, src)
    }
}

pub struct Weights {
    nexp: usize,
    blks: Box<[LlamaBlkStorage<DevBlob>]>,
    output_norm: DevBlob,
    output: DevBlob,
}

impl Weights {
    pub fn new(model: &LlamaStorage<&[u8]>, dist: Distribution, dev: &Device) -> Self {
        let LlamaStorage {
            meta,
            output_norm,
            output,
            blocks,
            ..
        } = model;

        let meta = meta.distribute(dist);
        let blks = blocks
            .iter()
            .map(|blk| {
                blk.clone()
                    .into_vec()
                    .into_iter()
                    .map(|(which, data)| {
                        let host: Contiguous<'_, _> =
                            meta.distribute_data(which, data, dist, |len| {
                                dev.malloc_host::<u8>(len)
                            });
                        let mut blob = dev.malloc::<u8>(host.len());
                        dev.memcpy_h2d(&mut blob, &host);
                        (which, blob)
                    })
                    .collect::<LlamaBlkStorage<_>>()
            })
            .collect::<Vec<_>>();
        let output_norm = dev.from_host(output_norm);
        let output = dev.from_host(output);

        Self {
            nexp: meta.nexp,
            blks: blks.into_boxed_slice(),
            output_norm,
            output,
        }
    }
}

impl WeightLoader for Weights {
    type Hardware = Device;

    type Weight<'s>
        = &'s [DevByte]
    where
        Self: 's;

    fn load_blk<'a>(
        &'a self,
        which: LlamaBlkWeight,
        iblk: usize,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a> {
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

    fn output_norm<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a> {
        &self.output_norm
    }

    fn output<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a> {
        &self.output
    }
}

#[cfg(test)]
mod infer;
