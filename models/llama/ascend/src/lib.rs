#![cfg(hw_detected)]

use llama::{BlkWeight, Contiguous, LlamaBlkStorage, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    ascend::Npu,
    ascendcl::{memcpy_d2h, DevByte, DevMem, Event, HostMem, Stream},
    random_sample::ascend::Operator as RandomSampleNpu,
    rearrange::ascend::Operator as Rearrange,
    ByteOf, QueueOf, TopoNode,
};
use std::{
    marker::PhantomData,
    mem::replace,
    ops::{Deref, RangeBounds},
};

pub struct Operators<N = Npu, R = NonAllReduce<Npu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Npu, RandomSampleNpu>;

pub struct Weights<'ctx> {
    blks: Box<[LlamaBlkStorage<DevMem<'ctx>>]>,
    output_norm: DevMem<'ctx>,
    output: DevMem<'ctx>,
}

macro_rules! op {
    ($name:ident) => {
        operators::$name::ascend::Operator
    };
}

impl<N, R> llama::Operators for Operators<N, R>
where
    N: TopoNode<Npu>,
    R: AllReduce<Npu, N>,
{
    type Hardware = Npu;
    type TopoNode = N;
    type RmsNorm = op!(rms_norm);
    type MatMul = op!(mat_mul);
    type Rope = op!(rope);
    type AttnKVCached = op!(attention_kv_cached);
    type Mlp = op!(mlp);
    type Rearrange = op!(rearrange);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>, _queue: &QueueOf<Self::Hardware>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        let tensor = tensor.as_ref().map(|s| {
            let mut host = vec![0u8; s.len()];
            memcpy_d2h(&mut host, s);
            host
        });
        println!("{tensor}");
    }
}

impl<'blk> Weights<'blk> {
    pub fn new(
        model: &LlamaStorage<&'_ [u8]>,
        range: impl RangeBounds<usize> + Clone,
        count: usize,
        pool_size: usize,
        stream: &Stream<'blk>,
    ) -> Self {
        assert!(pool_size > 0);
        if pool_size < model.meta.nblk {
            todo!()
        } else {
            let mut loader = None;
            Self {
                blks: model
                    .blocks
                    .iter()
                    .map(|blk| {
                        let blk = blk.distribute(&model.meta, range.clone(), count, |len| {
                            stream.ctx().malloc_host::<u8>(len)
                        });
                        let loader = loader.get_or_insert_with(|| {
                            blk.as_ref().map(|s| H2DLoader::new(s.len(), stream))
                        });
                        macro_rules! load {
                            ($( $ident:ident )+ ) => {
                                LlamaBlkStorage{
                                    $( $ident: loader.$ident.load(blk.$ident, stream) ),+
                                }
                            };
                        }
                        load! {
                            attn_norm
                            attn_qkv
                            attn_o
                            ffn_norm
                            ffn_gate_up
                            ffn_down
                        }
                    })
                    .collect(),
                output_norm: stream.ctx().from_host(model.output_norm),
                output: stream.ctx().from_host(model.output),
            }
        }
    }
}

struct H2DLoader<'ctx> {
    event: Event<'ctx>,
    host: HostMem<'ctx>,
    dev: DevMem<'ctx>,
}

impl<'ctx> H2DLoader<'ctx> {
    fn new(size: usize, stream: &Stream<'ctx>) -> Self {
        Self {
            event: stream.record(),
            host: stream.ctx().malloc_host::<u8>(size),
            dev: stream.ctx().malloc::<u8>(size),
        }
    }

    fn load(&mut self, host: Contiguous<HostMem<'ctx>>, stream: &Stream<'ctx>) -> DevMem<'ctx> {
        self.event.synchronize();
        match host {
            Contiguous::Borrowed(host) => self.host.copy_from_slice(host),
            Contiguous::Owned(host) => self.host = host,
        };
        stream.memcpy_h2d(&mut self.dev, &self.host);
        self.event = stream.record();
        replace(&mut self.dev, stream.ctx().malloc::<u8>(self.host.len()))
    }
}

impl WeightLoader for Weights<'_> {
    type Hardware = Npu;
    type Weight<'s>
        = &'s [DevByte]
    where
        Self: 's;

    #[inline]
    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Weight<'_> {
        let blk = &self.blks[iblk];
        match which {
            BlkWeight::AttnNorm => &blk.attn_norm,
            BlkWeight::AttnQKV => &blk.attn_qkv,
            BlkWeight::AttnO => &blk.attn_o,
            BlkWeight::FfnNorm => &blk.ffn_norm,
            BlkWeight::FfnGateUp => &blk.ffn_gate_up,
            BlkWeight::FfnDown => &blk.ffn_down,
        }
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
mod test_infer;
