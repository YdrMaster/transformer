use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{f16, upos, utok, Blob, FileLoadError};
use common_cpu::{
    tensor::{reslice, slice, udim, Tensor},
    CpuKernels, Kernels, KernelsA, KernelsB, ThisThread,
};
use llama::{ComputeConst, ComputeStream, Handle, LayerStorage, QueueOf, SliceOn, Storage, Weight};
use std::{iter::repeat, ops::Deref, path::Path, slice::from_raw_parts};

pub struct Transformer {
    s: Storage,
    kernels: CpuKernels,
}

impl Model for Transformer {
    type Meta = ();
    type Error = FileLoadError;

    #[inline]
    fn load(model_dir: impl AsRef<Path>, _meta: Self::Meta) -> Result<Self, Self::Error> {
        Ok(Self {
            s: llama::Storage::load_safetensors(model_dir)?,
            kernels: Default::default(),
        })
    }
}

impl ComputeStream for Transformer {
    type Handle = common_cpu::Cpu;
    type Storage = Blob;
    type Buf<'m> = Blob;
    type Pos<'m> = &'m [u8];

    #[inline]
    fn malloc(&self, len: usize) -> Self::Buf<'_> {
        Blob::new(len)
    }
    #[inline]
    fn map_pos<'p>(&self, pos: &'p [u32]) -> Self::Pos<'p>
    where
        Self: 'p,
    {
        reslice(pos)
    }
    #[inline]
    fn map_storage<'a>(&'a self, storage: &'a mut Self::Storage) -> &'a mut SliceOn<Self::Handle> {
        storage
    }
    #[inline]
    fn kernels(&self) -> &impl Kernels<Self::Handle> {
        &self.kernels
    }
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Handle> {
        &ThisThread
    }
    #[inline]
    fn constant(&self) -> ComputeConst {
        ComputeConst {
            nh: self.s.config.nh,
            nkvh: self.s.config.nkvh,
            di: self.s.config.di,
            epsilon: self.s.config.epsilon,
            theta: self.s.config.theta,
        }
    }

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = SliceOn<Self::Handle>>,
    {
        println!("{tensor}");
    }

    fn sync(&self) {
        println!("sync");
    }

    #[inline]
    fn layers(
        &self,
    ) -> impl Iterator<Item = impl llama::LLamaLayer<Byte = <Self::Handle as Handle>::Byte>> {
        self.s.layers.iter().map(LlamaLayer)
    }
}

struct LlamaLayer<'a>(&'a LayerStorage<Weight>);

impl<'a> llama::LLamaLayer for LlamaLayer<'a> {
    type Byte = u8;
    type Storage<'m> = Weight where Self: 'm;

    #[inline]
    fn att_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        self.0.att_layernorm.clone()
    }
    #[inline]
    fn att_qkv(&self) -> Tensor<Self::Storage<'_>> {
        self.0.att_qkv.clone()
    }
    #[inline]
    fn att_o(&self) -> Tensor<Self::Storage<'_>> {
        self.0.att_o.clone()
    }
    #[inline]
    fn mlp_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        self.0.mlp_layernorm.clone()
    }
    #[inline]
    fn mlp_gate_up(&self) -> Tensor<Self::Storage<'_>> {
        self.0.mlp_gate_up.clone()
    }
    #[inline]
    fn mlp_down(&self) -> Tensor<Self::Storage<'_>> {
        self.0.mlp_down.clone()
    }
}

impl CausalLM for Transformer {
    type Storage = Blob;

    #[inline]
    fn max_seq_len(&self) -> upos {
        self.s.config.max_seq_len
    }
    #[inline]
    fn eos_token(&self) -> utok {
        self.s.config.eos_token
    }
    #[inline]
    fn new_cache(&self) -> Tensor<Self::Storage> {
        self.s.config.new_cache(Blob::new)
    }
    #[inline]
    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        self.s
            .config
            .duplicate_cache(cache, pos, Blob::new, |dst, src| {
                src.map_physical(|u| &**u)
                    .reform_to(&mut dst.map_physical(|u| &mut **u))
            })
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let dt = self.s.config.dt;
        let d = self.s.config.d;

        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;

        let mut x = Tensor::alloc(dt, &[nt, d], Blob::new);
        self.kernels
            .gather(&mut x, &self.s.embed_tokens, tokens, &ThisThread);
        x
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        <Self as ComputeStream>::forward(self, queries, token_embedded)
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let dt = self.s.config.dt;
        let d = self.s.config.d;
        let epsilon = self.s.config.epsilon;

        let mut x = hidden_state;
        let range = DecodingMeta::select(&mut x, decoding, |dst, src| dst.copy_from_slice(src));

        if range.is_empty() {
            return Tensor::alloc(dt, &[0, d as _], Blob::new);
        }

        let lm_layernorm = &self.s.lm_layernorm;
        let lm_head = &self.s.lm_head;
        let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
        let mut logits = Tensor::alloc(dt, &[x.shape()[0], lm_head.shape()[1]], Blob::new);

        // 复制一个 x 以实现原地归一化
        let x_ = x
            .as_ref()
            .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
        self.kernels()
            .rms_norm(&mut x, &x_, lm_layernorm, epsilon, self.queue());
        self.kernels()
            .mat_mul(&mut logits, 0., &x, lm_head, 1., self.queue());

        logits
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        let &[_, voc] = logits.shape() else { panic!() };
        let logits: &[f16] = reslice(logits.as_slice());
        args.into_iter()
            .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
            .enumerate()
            .map(|(i, args)| args.random(&common_cpu::slice!(logits; voc; [i])))
            .collect()
    }
}

#[test]
fn test_infer() {
    causal_lm::test_impl::<Transformer>(
        (),
        &[
            29966, 29989, 1792, 29989, 29958, 13, 29903, 388, 376, 18567, 29908, 304, 592, 21106,
            29879, 5299, 29989, 465, 22137, 29989, 29958, 13,
        ],
    );
}
