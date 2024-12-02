use gpt2::{
    ext::ggml_quants::{self, digit_layout::DigitLayout, f16, DataBlock, QuantExt},
    storage::{BlkStorage, Storage},
    BlkWeight, Contiguous, Tensor,
    TensorUsage::Computation,
    WeightLoader,
};

use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    common_cpu::{Cpu, ThisThread},
    random_sample::common_cpu::Operator as RandomSampleCpu,
    rearrange::common_cpu::Operator as Rearrange,
    Blob, ByteOf, QueueOf, TopoNode,
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

pub type RandomSample = gpt2::RandomSample<Cpu, RandomSampleCpu>;

pub struct Weights<'w> {
    blks: Box<[BlkStorage<Contiguous<'w, Blob>>]>,
    output_norm_weight: &'w [u8],
    output_norm_bias: &'w [u8],
    output: &'w [u8],
    pos_embd: &'w [u8],
    weight_cache: RefCell<WeightCache>,
    dt_embd: DigitLayout,
    dt_mat: DigitLayout,
    size_qkv: usize,
    size_o: usize,
    size_gate_up: usize,
    size_down: usize,
}

pub struct WeightCache {
    cache: Blob,
    cached_weight: BlkWeight,
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
    type LayerNorm = op!(layer_norm);
    type MatMul = op!(mat_mul);
    type AttnKVCached = op!(attention_kv_cached);
    type Rearrange = op!(rearrange);
    type AllReduce = R;
    type AddRows = op!(add_rows);
    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        println!("{tensor}");
    }
}

impl<'w> Weights<'w> {
    pub fn new(
        model: &'w Storage<&'w [u8]>,
        range: impl RangeBounds<usize> + Clone,
        count: usize,
    ) -> Self {
        let Storage {
            meta,
            output_norm_weight,
            output_norm_bias,
            output,
            blocks,
            pos_embd,
            ..
        } = model;

        let blks = blocks
            .iter()
            .map(|blk| blk.distribute(meta, range.clone(), count, Blob::new))
            .collect::<Box<_>>();
        let mut meta = meta.clone();
        meta.distribute(range.clone(), count);

        let size_qkv = meta.attn_qkv_weight(Computation).take();
        let size_o = meta.attn_o_weight(Computation).take();
        let size_gate_up = meta.ffn_up_weight(Computation).take();
        let size_down = meta.ffn_down_weight(Computation).take();

        let weight_cache = if meta.dt_embd == meta.dt_mat {
            RefCell::new(WeightCache {
                cache: Blob::new(0),
                cached_weight: BlkWeight::AttnQKVw,
                cached_weight_iblk: 0,
            })
        } else {
            let max_size = [size_qkv, size_o, size_gate_up + size_down]
                .into_iter()
                .max()
                .unwrap();
            let mut cache = Blob::new(max_size);
            dequant(
                meta.dt_mat,
                meta.dt_embd,
                &blks[0].attn_qkv_weight,
                &mut cache[..size_qkv],
            );

            RefCell::new(WeightCache {
                cache,
                cached_weight: BlkWeight::AttnQKVw,
                cached_weight_iblk: 0,
            })
        };

        Self {
            pos_embd,
            blks,
            output_norm_weight,
            output_norm_bias,
            output,
            weight_cache,
            dt_embd: meta.dt_embd,
            dt_mat: meta.dt_mat,
            size_qkv,
            size_o,
            size_gate_up,
            size_down,
        }
    }
}

pub enum Dequant<'s> {
    Cached(Ref<'s, WeightCache>, Range<usize>),
    Borrowed(&'s [u8]),
}

impl Deref for Dequant<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Cached(cache, range) => &cache.cache[range.clone()],
            Self::Borrowed(data) => data,
        }
    }
}

// return the dst_size, quant_cache can longer than dst_size
fn dequant(dt_src: DigitLayout, dt_tgt: DigitLayout, src: &[u8], tgt: &mut [u8]) {
    macro_rules! inner_case {
            ($dequant_ty:ty; $($quant_ty:ty),*) => {
                match dt_src {
                    $(
                        <$quant_ty>::ID => {
                            assert_eq!(src.len() % size_of::<$quant_ty>(), 0);
                            let src_len = src.len() / size_of::<$quant_ty>();
                            let dst_len = src_len * <$quant_ty>::COUNT;
                            assert_eq!(tgt.len(), dst_len * size_of::<$dequant_ty>());
                            let src = unsafe { from_raw_parts(src.as_ptr().cast::<$quant_ty>(), src_len) };
                            let dst = unsafe { from_raw_parts_mut(tgt.as_mut_ptr().cast::<$dequant_ty>(), dst_len) };
                            <$quant_ty>::dequantize_slice(dst, src).expect("dequant failed");
                        },
                    )*
                    _ => panic!("unsupported dequantization source"),
                }
            }
        }

    use ggml_quants::{Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1};
    assert!(dt_tgt != dt_src);
    match dt_tgt {
        f16::ID => inner_case!(f16; Q8_0, Q8_1, Q5_0, Q5_1, Q4_0, Q4_1),
        f32::ID => inner_case!(f32; Q8_0, Q8_1, Q5_0, Q5_1, Q4_0, Q4_1),
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
        let &Self {
            ref blks,
            ref weight_cache,
            dt_embd,
            dt_mat,
            size_qkv,
            size_o,
            size_gate_up,
            size_down,
            ..
        } = self;
        let BlkStorage {
            attn_norm_weight,
            attn_norm_bias,
            attn_qkv_weight,
            attn_qkv_bias,
            attn_output_weight,
            attn_output_bias,

            ffn_norm_weight,
            ffn_norm_bias,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
        } = &blks[iblk];

        use BlkWeight::{
            AttnNormb, AttnNormw, AttnOb, AttnOw, AttnQKVb, AttnQKVw, FfnDownb, FfnDownw, FfnNormb,
            FfnNormw, FfnUpb, FfnUpw,
        };
        use Dequant::{Borrowed, Cached};

        #[rustfmt::skip]
        match which {
            AttnNormw                       => return Borrowed(&attn_norm_weight  ),
            AttnNormb                       => return Borrowed(&attn_norm_bias    ),
            AttnQKVw                        => return Borrowed(&attn_qkv_weight   ),
            AttnQKVb                        => return Borrowed(&attn_qkv_bias     ),
            AttnOw                          => return Borrowed(&attn_output_weight),
            AttnOb                          => return Borrowed(&attn_output_bias  ),

            FfnNormw                        => return Borrowed(&ffn_norm_weight   ),
            FfnNormb                        => return Borrowed(&ffn_norm_bias     ),
            FfnUpw                          => return Borrowed(&ffn_up_weight     ),
            FfnUpb                          => return Borrowed(&ffn_up_bias       ),
            FfnDownw                        => return Borrowed(&ffn_down_weight   ),
            FfnDownb                        => return Borrowed(&ffn_down_bias     ),
            _ => {}
        }; // dt_mat == dt_embd == F32

        let current_which = weight_cache.borrow().cached_weight;
        let current_iblk = weight_cache.borrow().cached_weight_iblk;
        if iblk != current_iblk
            || match which {
                FfnUpw | FfnDownw => !matches!(current_which, FfnUpw | FfnDownw),
                _ => current_which != which,
            }
        {
            let mut weight_cache = weight_cache.borrow_mut();
            let WeightCache {
                cache,
                cached_weight,
                cached_weight_iblk,
            } = &mut *weight_cache;
            *cached_weight = which;
            *cached_weight_iblk = iblk;
            #[rustfmt::skip]
            match which {
                AttnQKVw => dequant(dt_mat, dt_embd, attn_qkv_weight   , &mut cache[..size_qkv]),
                AttnOw   => dequant(dt_mat, dt_embd, attn_output_weight     , &mut cache[..size_o  ]),
                FfnUpw | FfnDownw => {
                           dequant(dt_mat, dt_embd, ffn_up_weight, &mut cache[..size_gate_up]);
                           dequant(dt_mat, dt_embd, ffn_down_weight   , &mut cache[size_gate_up..][..size_down]);
                }
                AttnNormw | FfnNormw => unreachable!(),
                AttnNormb | AttnQKVb | AttnOb | FfnNormb | FfnUpb | FfnDownb => unreachable!(),
            };
        }

        Cached(
            weight_cache.borrow(),
            match which {
                AttnQKVw => 0..size_qkv,
                AttnOw => 0..size_o,
                FfnUpw => 0..size_gate_up,
                FfnDownw => size_gate_up..size_gate_up + size_down,
                AttnNormw | FfnNormw => unreachable!(),
                AttnNormb | AttnQKVb | AttnOb | FfnNormb | FfnUpb | FfnDownb => unreachable!(),
            },
        )
    }

    #[inline]
    fn output_norm_weight(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        Dequant::Borrowed(self.output_norm_weight)
    }

    #[inline]
    fn output_norm_bias(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        Dequant::Borrowed(self.output_norm_bias)
    }
    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        Dequant::Borrowed(self.output)
    }
    #[inline]
    fn pos_embd<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a> {
        Dequant::Borrowed(self.pos_embd)
    }
}

#[cfg(test)]
pub mod test_infer;
