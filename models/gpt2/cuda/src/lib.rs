#![cfg(any(use_nvidia, use_iluvatar))]
use common::Slab;
use gpt2::{
    storage::{BlkStorage, Storage},
    BlkWeight, Contiguous, Tensor, WeightLoader,
};
use log::trace;
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    cuda::{memcpy_d2h, AsRaw, CurrentCtx, DevByte, DevMem, Event, Gpu, HostMem, Stream},
    random_sample::cuda::Operator as RandomSampleGpu,
    rearrange::cuda::Operator as Rearrange,
    ByteOf, QueueOf, TopoNode,
};
use std::ops::Deref;
use std::{marker::PhantomData, mem::replace, rc::Rc, time::Instant};

pub struct Operators<N = Gpu, R = NonAllReduce<Gpu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = gpt2::RandomSample<Gpu, RandomSampleGpu>;

pub struct Weights<'ctx> {
    blks: BlkStorage<Box<[DevMem<'ctx>]>>,
    output_norm_w: DevMem<'ctx>,
    output_norm_b: DevMem<'ctx>,
    output: DevMem<'ctx>,
    pos_embd: DevMem<'ctx>,
}

macro_rules! op {
    ($name:ident) => {
        operators::$name::cuda::Operator
    };
}

impl<N, R> gpt2::Operators for Operators<N, R>
where
    N: TopoNode<Gpu>,
    R: AllReduce<Gpu, N>,
{
    type Hardware = Gpu;
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
        let tensor = tensor.as_ref().map(|s| {
            let mut host = vec![0u8; s.len()];
            memcpy_d2h(&mut host, s);
            host
        });
        println!("{tensor}")
    }
    fn memcpy_d2h<T: Copy>(
        dst: &mut [T],
        src: &[ByteOf<Self::Hardware>],
        _queue: &QueueOf<Self::Hardware>,
    ) {
        memcpy_d2h(dst, src)
    }
}

impl<'blk> Weights<'blk> {
    pub fn new(model: &Storage<&'_ [u8]>, ctx: &'blk CurrentCtx) -> Self {
        let stream = Rc::new(ctx.stream());
        let igpu = unsafe { ctx.dev().as_raw() };
        let mut slab = Slab::new();
        let blks = {
            let mut loader = None;
            let mut blks_dev = model.blocks[0]
                .as_ref()
                .map(|_| Vec::with_capacity(model.meta.nblk));
            for (iblk, blk) in model.blocks.iter().enumerate() {
                let loader = loader
                    .get_or_insert_with(|| blk.as_ref().map(|s| H2DLoader::new(s.len(), &stream)));

                macro_rules! load {
                    ($( $ident:ident )+ ) => {
                        $(
                            let (dev, host) = loader.$ident.load(common::Contiguous::Borrowed(blk.$ident), &stream);
                            if let Some(host) = host {
                                slab.put(host.len(), host)
                            }
                            blks_dev.$ident.push(dev);
                        )+
                    };
                }
                let time = Instant::now();
                load! {
                    attn_qkv_b
                    attn_qkv_w
                    attn_o_b
                    attn_o_w
                    attn_norm_b
                    attn_norm_w
                    ffn_up_b
                    ffn_up_w
                    ffn_down_b
                    ffn_down_w
                    ffn_norm_b
                    ffn_norm_w
                }
                trace!("blk{iblk} loaded to gpu{igpu} in {:?}", time.elapsed())
            }
            blks_dev.map(|vec| vec.into_boxed_slice())
        };

        Self {
            pos_embd: ctx.from_host(model.pos_embd),
            blks,
            output_norm_w: ctx.from_host(model.output_norm_w),
            output_norm_b: ctx.from_host(model.output_norm_b),
            output: ctx.from_host(model.output),
        }
    }
}
impl WeightLoader for Weights<'_> {
    type Hardware = Gpu;
    type Memory<'s>
        = &'s [DevByte]
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
        } = &self.blks;

        match which {
            BlkWeight::AttnNorm => [&attn_norm_w[iblk], &attn_norm_b[iblk]],
            BlkWeight::AttnQKV => [&attn_qkv_w[iblk], &attn_qkv_b[iblk]],
            BlkWeight::AttnO => [&attn_o_w[iblk], &attn_o_b[iblk]],
            BlkWeight::FfnNorm => [&ffn_norm_w[iblk], &ffn_norm_b[iblk]],
            BlkWeight::FfnUp => [&ffn_up_w[iblk], &ffn_up_b[iblk]],
            BlkWeight::FfnDown => [&ffn_down_w[iblk], &ffn_down_b[iblk]],
        }
    }
    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> [Self::Memory<'_>; 2] {
        [&self.output_norm_w, &self.output_norm_b]
    }
    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        &self.output
    }
    #[inline]
    fn pos_embd<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a> {
        &self.pos_embd
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

    fn load(
        &mut self,
        host: Contiguous<HostMem<'ctx>>,
        stream: &Stream<'ctx>,
    ) -> (DevMem<'ctx>, Option<HostMem<'ctx>>) {
        self.event.synchronize();
        let cache = match host {
            Contiguous::Borrowed(host) => {
                self.host.copy_from_slice(host);
                None
            }
            Contiguous::Owned(host) => Some(replace(&mut self.host, host)),
        };
        stream.memcpy_h2d(&mut self.dev, &self.host);
        self.event = stream.record();
        (
            replace(&mut self.dev, stream.ctx().malloc::<u8>(self.host.len())),
            cache,
        )
    }
}
#[cfg(test)]
pub mod infer;
