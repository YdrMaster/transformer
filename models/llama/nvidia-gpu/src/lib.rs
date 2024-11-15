#![cfg(hw_detected)]

use llama::{BlkWeight, Contiguous, LlamaBlkStorage, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    cuda::{memcpy_d2h, DevByte, DevMem, Event, HostMem, Stream},
    nvidia_gpu::Gpu,
    random_sample::nvidia_gpu::Operator as RandomSampleGpu,
    rearrange::nvidia_gpu::Operator as Rearrange,
    ByteOf, QueueOf, TopoNode,
};
use std::{
    cell::{RefCell, RefMut},
    marker::PhantomData,
    mem::replace,
    ops::{Deref, RangeBounds},
};

pub struct Operators<N = Gpu, R = NonAllReduce<Gpu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Gpu, RandomSampleGpu>;

pub struct Weights<'ctx> {
    blks: Box<[LlamaBlkStorage<DevMem<'ctx>>]>,
    blks_roll_caches: LlamaBlkStorage<RefCell<RollCache<'ctx>>>,
    #[allow(dead_code)]
    blk_source: Box<[LlamaBlkStorage<HostMem<'ctx>>]>,
    output_norm: DevMem<'ctx>,
    output: DevMem<'ctx>,
    pool_size: usize,
    nblk: usize,
    stream: Stream<'ctx>,
}

pub struct RollCache<'ctx> {
    blk_idx: usize,
    start_idx: usize,
    cache: Box<[(DevMem<'ctx>, Event<'ctx>)]>,
}

impl<'ctx> RollCache<'ctx> {
    pub fn new(cache: Box<[(DevMem<'ctx>, Event<'ctx>)]>) -> Self {
        Self {
            blk_idx: 0,
            start_idx: 0,
            cache,
        }
    }

    pub fn first_event(&self) -> &Event<'ctx> {
        let (_, ref event) = self.cache[self.start_idx];
        event
    }
}

pub enum WeightResult<'s, 'ctx> {
    // (roll_cache,nblk,stream,blk_source)
    RollCached(
        RefMut<'s, RollCache<'ctx>>,
        usize,
        &'s Stream<'ctx>,
        &'s HostMem<'ctx>,
    ),
    Borrowed(&'s [DevByte]),
}

impl Deref for WeightResult<'_, '_> {
    type Target = [DevByte];

    fn deref(&self) -> &Self::Target {
        match self {
            WeightResult::RollCached(roll_cache, _, _, _) => {
                let (dev_mem, _event) = &roll_cache.cache[roll_cache.start_idx];

                dev_mem
            }
            WeightResult::Borrowed(dev_mem) => dev_mem,
        }
    }
}

impl Drop for WeightResult<'_, '_> {
    fn drop(&mut self) {
        match self {
            WeightResult::RollCached(roll_cache, nblk, stream, blk_source) => {
                roll_cache.blk_idx = (roll_cache.blk_idx + 1) % *nblk;

                let start_idx = roll_cache.start_idx;
                let (dev_mem, event) = &mut roll_cache.cache[start_idx];
                assert!(dev_mem.len() == blk_source.len());
                stream.memcpy_h2d(dev_mem, blk_source);
                *event = stream.record();

                roll_cache.start_idx = (roll_cache.start_idx + 1) % roll_cache.cache.len();
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
        stream: Stream<'blk>,
    ) -> Self {
        assert!(pool_size > 0);
        if pool_size < model.meta.nblk {
            let mut blks_roll_caches = model.blocks[0]
                .as_ref()
                .map(|_| Vec::with_capacity(pool_size));
            let blk_source = model
                .blocks
                .iter()
                .enumerate()
                .map(|(i, blk)| {
                    let blk = blk
                        .distribute(&model.meta, range.clone(), count, |len| {
                            stream.ctx().malloc_host::<u8>(len)
                        })
                        .map(|host| match host {
                            Contiguous::Borrowed(host) => {
                                let mut ans = stream.ctx().malloc_host::<u8>(host.len());
                                assert!(ans.len() == host.len());
                                ans.copy_from_slice(host);
                                ans
                            }
                            Contiguous::Owned(host) => host,
                        });
                    macro_rules! load {
                        ($( $ident:ident )+ ) => {
                                $( {blks_roll_caches.$ident.push(
                                            (stream.from_host(&blk.$ident),stream.record())
                                        );
                                    }
                                )+

                        };
                    }
                    if i < pool_size {
                        load! {
                            attn_norm
                            attn_qkv
                            attn_o
                            ffn_norm
                            ffn_gate_up
                            ffn_down
                        }
                    }
                    blk
                })
                .collect::<Box<[_]>>();
            let blks_roll_caches =
                blks_roll_caches.map(|vec| RefCell::new(RollCache::new(vec.into())));

            Self {
                blks: Box::new([]),
                blks_roll_caches,
                blk_source,
                output_norm: stream.from_host(model.output_norm),
                output: stream.from_host(model.output),
                pool_size,
                nblk: model.meta.nblk,
                stream,
            }
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
                            blk.as_ref().map(|s| H2DLoader::new(s.len(), &stream))
                        });
                        macro_rules! load {
                            ($( $ident:ident )+ ) => {
                                LlamaBlkStorage{
                                    $( $ident: loader.$ident.load(blk.$ident, &stream) ),+
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
                blks_roll_caches: model.blocks[0]
                    .as_ref()
                    .map(|_| RefCell::new(RollCache::new(Box::new([])))),
                blk_source: Box::new([]),
                output_norm: stream.from_host(model.output_norm),
                output: stream.from_host(model.output),
                pool_size,
                nblk: model.meta.nblk,
                stream,
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
            dev: stream.malloc::<u8>(size),
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
        replace(&mut self.dev, stream.malloc::<u8>(self.host.len()))
    }
}

impl<'ctx> WeightLoader for Weights<'ctx> {
    type Hardware = Gpu;
    type Memory<'s>
        = WeightResult<'s, 'ctx>
    where
        Self: 's;

    #[inline]
    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        queue: &QueueOf<Self::Hardware>,
    ) -> Self::Memory<'_> {
        assert!(iblk < self.nblk);
        if self.pool_size < self.nblk {
            macro_rules! cases {
                ($( $ty:ident,$ident:ident )+ ) => {
                    match which {
                        $(BlkWeight::$ty => {
                            let roll_cache = self.blks_roll_caches.$ident.borrow_mut();
                            queue.wait_for(roll_cache.first_event());
                            assert!(iblk == roll_cache.blk_idx);
                            let next_load_idx = (iblk + self.pool_size) % self.nblk;
                            let blk = &self.blk_source[next_load_idx].$ident;
                            WeightResult::RollCached(roll_cache, self.nblk, &self.stream, blk)
                        })+
                    }
                }
            }

            cases!(
                AttnNorm,attn_norm
                AttnQKV,attn_qkv
                AttnO,attn_o
                FfnNorm,ffn_norm
                FfnGateUp,ffn_gate_up
                FfnDown,ffn_down
            )
        } else {
            let blk = &self.blks[iblk];
            WeightResult::Borrowed(match which {
                BlkWeight::AttnNorm => &blk.attn_norm,
                BlkWeight::AttnQKV => &blk.attn_qkv,
                BlkWeight::AttnO => &blk.attn_o,
                BlkWeight::FfnNorm => &blk.ffn_norm,
                BlkWeight::FfnGateUp => &blk.ffn_gate_up,
                BlkWeight::FfnDown => &blk.ffn_down,
            })
        }
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        WeightResult::Borrowed(&self.output_norm)
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        WeightResult::Borrowed(&self.output)
    }
}

#[cfg(test)]
mod test_infer;
