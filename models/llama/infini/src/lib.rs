#![cfg(detected)]

use llama::{BlkWeight, Contiguous, LlamaBlkStorage, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    infini_rt::{DevBlob, DevByte, Device, Event, HostBlob, Stream},
    random_sample::infini::Operator as RandomSampleNpu,
    rearrange::infini::Operator as Rearrange,
    ByteOf, QueueOf, TopoNode,
};
use std::{
    marker::PhantomData,
    mem::replace,
    ops::{Deref, RangeBounds},
};

pub struct Operators<N = Device, R = NonAllReduce<Device, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Device, RandomSampleNpu>;

pub struct Weights {
    blks: Box<[LlamaBlkStorage<DevBlob>]>,
    output_norm: DevBlob,
    output: DevBlob,
}

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
    type Mlp = op!(mlp);
    type Rearrange = op!(rearrange);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>, queue: &QueueOf<Self::Hardware>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        let tensor = tensor.as_ref().map(|s| {
            let mut host = vec![0u8; s.len()];
            queue.get_device().memcpy_d2h(&mut host, s);
            host
        });
        println!("{tensor}");
    }
}

impl Weights {
    pub fn new(
        model: &LlamaStorage<&'_ [u8]>,
        range: impl RangeBounds<usize> + Clone,
        count: usize,
        stream: &Stream,
    ) -> Self {
        let device = stream.get_device();
        let mut loader = None;
        Self {
            blks: model
                .blocks
                .iter()
                .map(|blk| {
                    let blk = blk.distribute(&model.meta, range.clone(), count, |len| {
                        device.malloc_host::<u8>(len)
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
            output_norm: device.from_host(model.output_norm),
            output: device.from_host(model.output),
        }
    }
}

struct H2DLoader {
    event: Event,
    host: HostBlob,
    dev: DevBlob,
}

impl H2DLoader {
    fn new(size: usize, stream: &Stream) -> Self {
        let device = stream.get_device();
        let mut event = device.event();
        stream.record(&mut event);
        Self {
            event,
            host: device.malloc_host::<u8>(size),
            dev: device.malloc::<u8>(size),
        }
    }

    fn load(&mut self, host: Contiguous<HostBlob>, stream: &Stream) -> DevBlob {
        let device = stream.get_device();
        self.event.synchronize();
        match host {
            Contiguous::Borrowed(host) => self.host.copy_from_slice(host),
            Contiguous::Owned(host) => self.host = host,
        };
        device.memcpy_h2d(&mut self.dev, &self.host);
        stream.record(&mut self.event);
        replace(&mut self.dev, stream.malloc::<u8>(self.host.len()))
    }
}

impl WeightLoader for Weights {
    type Hardware = Device;
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
mod infer;
