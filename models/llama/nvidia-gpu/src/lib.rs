#![cfg(driver_detected)]

use llama::{BlkWeight, Contiguous, LlamaBlkStorage, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    cuda::{memcpy_d2h, CurrentCtx, DevByte, DevMem, Event, HostMem, Stream},
    nvidia_gpu::Gpu,
    random_sample::nvidia_gpu::Operator as RandomSampleGpu,
    rearrange::nvidia_gpu::Operator as Rearrange,
    Blob, ByteOf, QueueOf, TopoNode,
};
use std::{
    cell::{RefCell, RefMut},
    marker::PhantomData,
    mem::replace,
    ops::{Deref, RangeBounds},
    rc::Rc,
};

pub struct Operators<N = Gpu, R = NonAllReduce<Gpu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Gpu, RandomSampleGpu>;

pub struct Weights<'ctx> {
    blks: LlamaBlkStorage<Cache<'ctx>>,
    output_norm: DevMem<'ctx>,
    output: DevMem<'ctx>,
}

pub enum Cache<'ctx> {
    Static(Box<[DevMem<'ctx>]>),
    Rolling {
        stream: Rc<Stream<'ctx>>,
        host: Box<[HostMem<'ctx>]>,
        dev: RefCell<RollCache<'ctx>>,
    },
}

pub struct RollCache<'ctx> {
    global_idx: usize,
    local_idx: usize,
    nblk: usize,
    cache: Box<[(DevMem<'ctx>, Event<'ctx>)]>,
}

impl<'ctx> RollCache<'ctx> {
    pub fn new(nblk: usize, cache: Box<[(DevMem<'ctx>, Event<'ctx>)]>) -> Self {
        Self {
            global_idx: 0,
            local_idx: 0,
            nblk,
            cache,
        }
    }

    pub fn first_event(&self) -> &Event<'ctx> {
        let (_, ref event) = self.cache[self.local_idx];
        event
    }
}

pub enum WeightResult<'s, 'ctx> {
    RollCached {
        roll_cache: RefMut<'s, RollCache<'ctx>>,
        load_stream: &'s Stream<'ctx>,
        host: &'s [HostMem<'ctx>],
        compute_stream: &'s Stream<'s>,
    },
    Borrowed(&'s [DevByte]),
}

impl Deref for WeightResult<'_, '_> {
    type Target = [DevByte];

    fn deref(&self) -> &Self::Target {
        match self {
            WeightResult::RollCached { roll_cache, .. } => {
                &roll_cache.cache[roll_cache.local_idx].0
            }
            WeightResult::Borrowed(dev_mem) => dev_mem,
        }
    }
}

impl Drop for WeightResult<'_, '_> {
    fn drop(&mut self) {
        match self {
            WeightResult::RollCached {
                roll_cache,
                load_stream,
                host,
                compute_stream,
            } => {
                // wait for the compute to finish
                load_stream.wait_for(&compute_stream.record());

                let next_load_idx =
                    (roll_cache.global_idx + roll_cache.cache.len()) % roll_cache.nblk;
                let host = &host[next_load_idx];

                roll_cache.global_idx = (roll_cache.global_idx + 1) % roll_cache.nblk;

                let start_idx = roll_cache.local_idx;
                let (dev_mem, event) = &mut roll_cache.cache[start_idx];
                assert!(dev_mem.len() == host.len());
                load_stream.memcpy_h2d(dev_mem, host);
                *event = load_stream.record();

                roll_cache.local_idx = (roll_cache.local_idx + 1) % roll_cache.cache.len();
            }
            WeightResult::Borrowed(_) => {}
        }
    }
}

macro_rules! op {
    ($name:ident) => {
        operators::$name::nvidia_gpu::Operator
    };
}

impl<N, R> llama::Operators for Operators<N, R>
where
    N: TopoNode<Gpu>,
    R: AllReduce<Gpu, N>,
{
    type Hardware = Gpu;
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
        ctx: &'blk CurrentCtx,
    ) -> Self {
        assert!(pool_size > 0);
        let stream = Rc::new(ctx.stream());
        let blks = if pool_size < model.meta.nblk {
            let mut blks_host = model.blocks[0]
                .as_ref()
                .map(|_| Vec::with_capacity(model.meta.nblk));
            for blk in model.blocks.iter() {
                let blk = blk
                    .distribute(&model.meta, range.clone(), count, |len| {
                        ctx.malloc_host::<u8>(len)
                    })
                    .map(|host| match host {
                        Contiguous::Borrowed(host) => {
                            let mut ans = ctx.malloc_host::<u8>(host.len());
                            ans.copy_from_slice(host);
                            ans
                        }
                        Contiguous::Owned(host) => host,
                    });

                macro_rules! push {
                    ($( $ident:ident )+ ) => {
                        $({ blks_host.$ident.push(blk.$ident); })+
                    };
                }
                push! {
                    attn_norm
                    attn_qkv
                    attn_o
                    ffn_norm
                    ffn_gate_up
                    ffn_down
                }
            }
            blks_host.map(|vec| {
                let roll_cache = vec
                    .iter()
                    .take(pool_size)
                    .map(|host| (stream.from_host(host), stream.record()))
                    .collect::<Box<_>>();
                Cache::Rolling {
                    stream: stream.clone(),
                    host: vec.into_boxed_slice(),
                    dev: RefCell::new(RollCache::new(model.meta.nblk, roll_cache)),
                }
            })
        } else {
            let mut loader = None;
            let mut blks_dev = model.blocks[0]
                .as_ref()
                .map(|_| Vec::with_capacity(model.meta.nblk));
            for blk in &model.blocks {
                let blk = blk.distribute(&model.meta, range.clone(), count, Blob::new);
                let loader = loader
                    .get_or_insert_with(|| blk.as_ref().map(|s| H2DLoader::new(s.len(), &stream)));

                macro_rules! load {
                    ($( $ident:ident )+ ) => {
                        $({ blks_dev.$ident.push(loader.$ident.load(blk.$ident, &stream)); })+
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
            }
            blks_dev.map(|vec| Cache::Static(vec.into_boxed_slice()))
        };

        Self {
            blks,
            output_norm: stream.from_host(model.output_norm),
            output: stream.from_host(model.output),
        }
    }
}

struct H2DLoader<'ctx> {
    event: Event<'ctx>,
    host: Blob,
    dev: DevMem<'ctx>,
}

impl<'ctx> H2DLoader<'ctx> {
    fn new(size: usize, stream: &Stream<'ctx>) -> Self {
        Self {
            event: stream.record(),
            host: Blob::new(size),
            dev: stream.malloc::<u8>(size),
        }
    }

    fn load(&mut self, host: Contiguous<Blob>, stream: &Stream<'ctx>) -> DevMem<'ctx> {
        self.event.synchronize();
        match host {
            Contiguous::Borrowed(host) => self.host.copy_from_slice(host),
            Contiguous::Owned(host) => self.host = host,
        };
        stream.memcpy_h2d(&mut self.dev, &self.host);
        self.event = stream.record();
        replace(&mut self.dev, stream.malloc::<u8>(self.host.len()))
    }
}

impl<'ctx> WeightLoader for Weights<'ctx> {
    type Hardware = Gpu;
    type Weight<'s>
        = WeightResult<'s, 'ctx>
    where
        Self: 's;

    #[inline]
    fn load_blk<'s>(
        &'s self,
        which: BlkWeight,
        iblk: usize,
        queue: &'s QueueOf<Self::Hardware>,
    ) -> Self::Weight<'s> {
        let cache = match which {
            BlkWeight::AttnNorm => &self.blks.attn_norm,
            BlkWeight::AttnQKV => &self.blks.attn_qkv,
            BlkWeight::AttnO => &self.blks.attn_o,
            BlkWeight::FfnNorm => &self.blks.ffn_norm,
            BlkWeight::FfnGateUp => &self.blks.ffn_gate_up,
            BlkWeight::FfnDown => &self.blks.ffn_down,
        };

        match cache {
            Cache::Static(dev) => WeightResult::Borrowed(&dev[iblk]),
            Cache::Rolling { stream, host, dev } => {
                let roll_cache = dev.borrow_mut();
                queue.wait_for(roll_cache.first_event());
                assert!(iblk == roll_cache.global_idx);
                WeightResult::RollCached {
                    roll_cache,
                    load_stream: stream,
                    host,
                    compute_stream: queue,
                }
            }
        }
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        WeightResult::Borrowed(&self.output_norm)
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        WeightResult::Borrowed(&self.output)
    }
}

#[cfg(test)]
mod infer;

#[cfg(all(test, nccl_detected))]
mod nccl_parallel;
