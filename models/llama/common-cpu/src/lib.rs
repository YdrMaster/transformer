use common::Contiguous;
use llama::{
    ext::ggml_quants::{self, digit_layout::DigitLayout, f16, DataBlock, QuantExt},
    BlkWeight, LlamaBlkStorage, LlamaStorage, Tensor,
    TensorUsage::Computation,
    WeightLoader,
};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    common_cpu::Cpu,
    random_sample::common_cpu::Operator as RandomSampleCpu,
    rearrange::common_cpu::Operator as Rearrange,
    Blob, ByteOf, QueueOf, TopoNode,
};
use std::{
    cell::{Ref, RefCell},
    marker::PhantomData,
    mem::size_of,
    ops::{Deref, Range, RangeBounds},
    ptr::copy_nonoverlapping,
    slice::{from_raw_parts, from_raw_parts_mut},
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
    nexp: usize,
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
    type Swiglu = op!(swiglu);
    type Rearrange = op!(rearrange);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>, _queue: &QueueOf<Self::Hardware>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        println!("{tensor}")
    }

    fn memcpy_d2h<T: Copy>(
        dst: &mut [T],
        src: &[ByteOf<Self::Hardware>],
        _queue: &QueueOf<Self::Hardware>,
    ) {
        let count = size_of_val(dst);
        assert_eq!(size_of_val(src), count);
        unsafe { copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().cast::<u8>(), count) }
    }
}

impl<'w> Weights<'w> {
    pub fn new(
        model: &'w LlamaStorage<&'w [u8]>,
        range: impl RangeBounds<usize> + Clone,
        count: usize,
    ) -> Self {
        let LlamaStorage {
            meta,
            output_norm,
            output,
            blocks,
            ..
        } = model;

        let blks = blocks
            .iter()
            .map(|blk| blk.distribute(meta, range.clone(), count, Blob::new))
            .collect::<Box<_>>();

        let mut meta = meta.clone();
        meta.distribute(range.clone(), count);
        let size_qkv = meta.attn_qkv(Computation).take();
        let size_o = meta.attn_o(Computation).take();
        let size_gate_up = meta.ffn_gate_up(Computation).take();
        let size_down = meta.ffn_down(Computation).take();

        let weight_cache = if meta.dt_embd == meta.dt_mat {
            RefCell::new(WeightCache {
                cache: Blob::new(0),
                cached_weight: BlkWeight::AttnQKV,
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
                &blks[0].attn_qkv,
                &mut cache[..size_qkv],
            );

            RefCell::new(WeightCache {
                cache,
                cached_weight: BlkWeight::AttnQKV,
                cached_weight_iblk: 0,
            })
        };
        Self {
            blks,
            output_norm,
            output,
            weight_cache,
            dt_embd: meta.dt_embd,
            dt_mat: meta.dt_mat,
            nexp: meta.nexp,
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
    type Weight<'s>
        = Dequant<'s>
    where
        Self: 's;

    #[inline]
    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Weight<'_> {
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
        let LlamaBlkStorage {
            attn_norm,
            attn_qkv,
            attn_o,
            ffn_norm,
            ffn_gate_inp,
            ffn_gate_up,
            ffn_down,
        } = &blks[iblk];

        use BlkWeight::{AttnNorm, AttnO, AttnQKV, FfnDown, FfnGateInp, FfnGateUp, FfnNorm};
        use Dequant::{Borrowed, Cached};

        #[rustfmt::skip]
        match which {
            AttnNorm                        => return Borrowed(attn_norm   ),
            AttnQKV    if dt_mat == dt_embd => return Borrowed(attn_qkv    ),
            AttnO      if dt_mat == dt_embd => return Borrowed(attn_o      ),
            FfnNorm                         => return Borrowed(ffn_norm    ),
            FfnGateInp if dt_mat == dt_embd => return Borrowed(ffn_gate_inp),
            FfnGateUp  if dt_mat == dt_embd => return Borrowed(ffn_gate_up ),
            FfnDown    if dt_mat == dt_embd => return Borrowed(ffn_down    ),
            _ => {}
        };

        let current_which = weight_cache.borrow().cached_weight;
        let current_iblk = weight_cache.borrow().cached_weight_iblk;
        if iblk != current_iblk
            || match which {
                FfnGateUp | FfnDown => !matches!(current_which, FfnGateUp | FfnDown),
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
            match which {
                AttnQKV => dequant(dt_mat, dt_embd, attn_qkv, &mut cache[..size_qkv]),
                AttnO => dequant(dt_mat, dt_embd, attn_o, &mut cache[..size_o]),
                FfnGateInp => todo!("dequant ffn gate inp"),
                FfnGateUp | FfnDown => {
                    dequant(dt_mat, dt_embd, ffn_gate_up, &mut cache[..size_gate_up]);
                    dequant(
                        dt_mat,
                        dt_embd,
                        ffn_down,
                        &mut cache[size_gate_up..][..size_down],
                    );
                }
                AttnNorm | FfnNorm => unreachable!(),
            };
        }

        Cached(
            weight_cache.borrow(),
            match which {
                AttnQKV => 0..size_qkv,
                AttnO => 0..size_o,
                FfnGateInp => todo!("dequant ffn gate inp"),
                FfnGateUp => 0..size_gate_up,
                FfnDown => size_gate_up..size_gate_up + size_down,
                AttnNorm | FfnNorm => unreachable!(),
            },
        )
    }

    fn load_moe<'a>(
        &'a self,
        which: BlkWeight,
        iblk: usize,
        iexp: usize,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a> {
        let &Self {
            ref blks,
            dt_embd,
            dt_mat,
            nexp,
            ..
        } = self;
        assert_eq!(dt_embd, dt_mat);
        let w = match which {
            BlkWeight::FfnGateUp => &*blks[iblk].ffn_gate_up,
            BlkWeight::FfnDown => &*blks[iblk].ffn_down,
            _ => unreachable!(),
        };
        let one = w.len() / nexp;
        Dequant::Borrowed(&w[iexp * one..][..one])
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        Dequant::Borrowed(self.output_norm)
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        Dequant::Borrowed(self.output)
    }
}

#[cfg(test)]
mod infer;
