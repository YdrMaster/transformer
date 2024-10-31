use llama::ext::{
    f16,
    ggml_quants::{self, DataBlock, QuantExt},
    DigitLayout,
};
use llama::{
    BlkWeight, Contiguous, LlamaBlkStorage, LlamaStorage, Tensor, TensorUsage::Computation,
    WeightLoader,
};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    common_cpu::{Blob, Cpu},
    random_sample::common_cpu::Operator as RandomSampleCpu,
    rearrange::common_cpu::Operator as Rearrange,
    ByteOf, QueueOf, TopoNode,
};
use std::{
    cell::Ref,
    ops::Range,
    slice::{from_raw_parts, from_raw_parts_mut},
};
use std::{
    cell::RefCell,
    marker::PhantomData,
    mem::size_of,
    ops::{Deref, RangeBounds},
};

pub struct Operators<N = Cpu, R = NonAllReduce<Cpu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Cpu, RandomSampleCpu>;

pub struct Weights<'w> {
    blks: Box<[LlamaBlkStorage<Contiguous<'w, Blob>>]>,
    output_norm: &'w [u8],
    output: &'w [u8],
    weight_cache: RefCell<WeightCache>,
    dt_embd: DigitLayout,
    dt_mat: DigitLayout,
}

pub struct WeightCache {
    cache: Blob,
    cached_weight: BlkWeight,
    cached_weight_iblk: usize,
    used: Range<usize>,
}
macro_rules! op {
    ($name:ident) => {
        operators::$name::common_cpu::Operator
    };
}

impl<N, R> llama::Operators for Operators<N, R>
where
    N: TopoNode<Cpu>,
    R: AllReduce<Cpu, N>,
{
    type Hardware = Cpu;
    type TopoNode = N;
    type RmsNorm = op!(rms_norm);
    type MatMul = op!(mat_mul);
    type Rope = op!(rope);
    type AttnKVCached = op!(attention_kv_cached);
    type Mlp = op!(mlp);
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
    pub fn new(
        model: &'w LlamaStorage<&'w [u8]>,
        range: impl RangeBounds<usize> + Clone,
        count: usize,
    ) -> Self {
        let LlamaStorage {
            output_norm,
            output,
            blocks,
            ..
        } = model;
        let weight_cache = if model.meta.dt_embd == model.meta.dt_mat {
            RefCell::new(WeightCache {
                cache: Blob::new(0),
                cached_weight: BlkWeight::AttnQKV,
                cached_weight_iblk: 0,
                used: 0..0,
            })
        } else {
            let max_size = [
                model.meta.attn_qkv(Computation).take(),
                model.meta.attn_o(Computation).take(),
                model.meta.ffn_gate_up(Computation).take()
                    + model.meta.ffn_down(Computation).take(),
            ]
            .into_iter()
            .max()
            .unwrap();
            let mut cache = Blob::new(max_size);
            let cache_used = dequant_data(
                model.meta.dt_mat,
                model.meta.dt_embd,
                blocks[0].attn_qkv,
                &mut cache,
            );

            RefCell::new(WeightCache {
                cache,
                cached_weight: BlkWeight::AttnQKV,
                cached_weight_iblk: 0,
                used: 0..cache_used,
            })
        };
        Self {
            blks: blocks
                .iter()
                .map(|blk| blk.distribute(&model.meta, range.clone(), count, Blob::new))
                .collect(),
            output_norm,
            output,
            weight_cache,
            dt_embd: model.meta.dt_embd,
            dt_mat: model.meta.dt_mat,
        }
    }
}

pub enum Dequant<'s> {
    Buffered(Ref<'s, WeightCache>, Range<usize>),
    Borrowed(&'s [u8]),
}

impl Deref for Dequant<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Dequant::Buffered(cache, Range { start, end }) => &cache.cache[*start..*end],
            Dequant::Borrowed(data) => data,
        }
    }
}

// return the dst_size, quant_cache can longer than dst_size
fn dequant_data(
    dt_mat: DigitLayout,
    dt_embd: DigitLayout,
    data: &[u8],
    quant_cache: &mut [u8],
) -> usize {
    macro_rules! inner_case {
            ($dequant_ty:ty,$($quant_ty:ty),*) => {
                match dt_mat {
                    $(
                        <$quant_ty>::ID => {
                            assert!(data.len() % size_of::<$quant_ty>() == 0);
                            let src_len = data.len() / size_of::<$quant_ty>();
                            let dst_len = src_len * <$quant_ty>::COUNT;
                            assert!(quant_cache.len() >= dst_len * size_of::<$dequant_ty>());
                            let src = unsafe {
                                from_raw_parts(data.as_ptr().cast::<$quant_ty>(), src_len)
                            };
                            let dst = unsafe {
                                from_raw_parts_mut(quant_cache.as_mut_ptr().cast::<$dequant_ty>(), dst_len)
                            };
                            <$quant_ty>::dequantize_slice(dst, src).expect("dequant failed");
                            dst_len * size_of::<$dequant_ty>()
                        },
                    )*
                    _ => panic!("unsupported dequantization source"),
                }
            }
        }

    assert!(dt_embd != dt_mat);
    match dt_embd {
        f16::ID => inner_case!(f16, ggml_quants::Q8_0),
        f32::ID => inner_case!(f32, ggml_quants::Q8_0),
        _ => panic!("unsupported dequantization target"),
    }
}

impl WeightLoader for Weights<'_> {
    type Hardware = Cpu;
    type Memory<'s>
        = Dequant<'s>
    where
        Self: 's;

    #[inline]
    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Memory<'_> {
        let blk = &self.blks[iblk];

        let mut is_mat = false;
        let data = match which {
            BlkWeight::AttnQKV => {
                is_mat = true;
                &blk.attn_qkv
            }
            BlkWeight::AttnO => {
                is_mat = true;
                &blk.attn_o
            }
            BlkWeight::FfnGateUp => {
                is_mat = true;
                &blk.ffn_gate_up
            }
            BlkWeight::FfnDown => {
                is_mat = true;
                &blk.ffn_down
            }
            BlkWeight::FfnNorm => &blk.ffn_norm,
            BlkWeight::AttnNorm => &blk.attn_norm,
        };
        let cached_weight = self.weight_cache.borrow().cached_weight;
        let cached_weight_iblk = self.weight_cache.borrow().cached_weight_iblk;
        if is_mat && (self.dt_mat != self.dt_embd) {
            match which {
                BlkWeight::AttnQKV | BlkWeight::AttnO => {
                    if which != cached_weight || iblk != cached_weight_iblk {
                        let mut weight_cache = self.weight_cache.borrow_mut();
                        let used =
                            dequant_data(self.dt_mat, self.dt_embd, data, &mut weight_cache.cache);
                        weight_cache.cached_weight = which;
                        weight_cache.cached_weight_iblk = iblk;
                        weight_cache.used = 0..used;
                    }
                    Dequant::Buffered(
                        self.weight_cache.borrow(),
                        self.weight_cache.borrow().used.clone(),
                    )
                }
                BlkWeight::FfnGateUp | BlkWeight::FfnDown => {
                    if !(cached_weight == which
                        || (cached_weight == BlkWeight::FfnGateUp && which == BlkWeight::FfnDown)
                        || (cached_weight == BlkWeight::FfnDown && which == BlkWeight::FfnGateUp))
                        || iblk != cached_weight_iblk
                    {
                        let mut weight_cache = self.weight_cache.borrow_mut();
                        let used1 = dequant_data(
                            self.dt_mat,
                            self.dt_embd,
                            &blk.ffn_gate_up,
                            &mut weight_cache.cache,
                        );
                        let used2 = dequant_data(
                            self.dt_mat,
                            self.dt_embd,
                            &blk.ffn_down,
                            &mut weight_cache.cache[used1..],
                        );
                        weight_cache.cached_weight = which;
                        weight_cache.cached_weight_iblk = iblk;
                        weight_cache.used = used1..used1 + used2;
                    }
                    match which {
                        BlkWeight::FfnGateUp => Dequant::Buffered(
                            self.weight_cache.borrow(),
                            0..self.weight_cache.borrow().used.start,
                        ),
                        BlkWeight::FfnDown => Dequant::Buffered(
                            self.weight_cache.borrow(),
                            self.weight_cache.borrow().used.clone(),
                        ),
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }
        } else {
            Dequant::Borrowed(data)
        }
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        Dequant::Borrowed(self.output_norm)
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        Dequant::Borrowed(self.output)
    }
}

#[cfg(test)]
mod test_infer;

#[cfg(test)]
mod test_dist;
