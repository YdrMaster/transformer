#![cfg(detected_cuda)]

mod resource;

#[macro_use]
extern crate log;

use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{upos, utok, Blob, FileLoadError};
use common_nv::{
    cuda::{memcpy_d2h, AsRaw},
    slice, udim, Gpu, Kernels, KernelsA, KernelsB, NvidiaKernels, Tensor,
};
use cuda::{
    ContextResource, ContextSpore, DevByte, DevMem, DevMemSpore, Device, EventSpore, HostMemSpore,
    Stream, StreamSpore,
};
use llama::{ComputeConst, InferenceConfig, LayerStorage, SliceOn, Weight};
use resource::Resource;
use std::{
    cell::RefCell,
    collections::VecDeque,
    iter::repeat,
    mem::{take, ManuallyDrop},
    ops::Deref,
    path::Path,
    rc::Rc,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

pub use common_nv::{cuda, synchronize};
pub use resource::Cache;

pub struct Transformer(ManuallyDrop<Internal>);

struct Internal {
    config: InferenceConfig,

    resource: Arc<Resource>,
    transfer: StreamSpore,
    kernels: NvidiaKernels,
    sample_workspace: DevMemSpore,

    embed_tokens: Tensor<HostMemSpore>,
    layers: Vec<LayerStorage<HostMemSpore>>,
    lm_layernorm: Tensor<DevMemSpore>,
    lm_head: Tensor<DevMemSpore>,

    pool: Mutex<VecDeque<(LayerStorage<DevMemSpore>, EventSpore)>>,
}

pub struct ModelLoadMeta {
    pub device: Device,
    pub load_layers: usize,
}

impl ModelLoadMeta {
    #[inline]
    pub fn load_all_to(n: i32) -> Self {
        Self {
            device: Device::new(n),
            load_layers: usize::MAX,
        }
    }
}

impl Model for Transformer {
    type Meta = ModelLoadMeta;
    type Error = FileLoadError;

    #[inline]
    fn load(
        model_dir: impl AsRef<Path>,
        Self::Meta {
            device,
            load_layers,
        }: Self::Meta,
    ) -> Result<Self, Self::Error> {
        let time = Instant::now();
        let host = llama::Storage::load_safetensors(model_dir)?;
        info!("load host: {:?}", time.elapsed());
        let load_layers = (load_layers as udim).min(host.config.nlayers);

        let resource = Arc::new(Resource::new(&device));
        device.set_mempool_threshold(u64::MAX);

        // 异步编译 CUDA
        let kernels = std::thread::spawn(move || {
            info!("compile CUDA kernels");
            NvidiaKernels::new(&[device], host.config.d as _, host.config.voc as _)
        });

        resource.apply(|compute| {
            let ctx = compute.ctx();
            let transfer = ctx.stream();

            let page_lock = |u: &Weight| {
                let mut host = ctx.malloc_host::<u8>(u.len());
                host.copy_from_slice(u);
                host.sporulate()
            };
            let from_host = |u: &HostMemSpore| transfer.from_host(u).sporulate();

            let layers = host
                .layers
                .iter()
                .map(|l| l.map(page_lock))
                .collect::<Vec<_>>();
            let pool = layers
                .iter()
                .take(load_layers as usize)
                .map(|l| (l.map(from_host), transfer.record().sporulate()))
                .collect();
            let embed_tokens = host.embed_tokens.as_ref().map_physical(page_lock);
            let lm_layernorm = host
                .lm_layernorm
                .map_physical(|u| transfer.from_host(&u).sporulate());
            let lm_head = host
                .lm_head
                .map_physical(|u| transfer.from_host(&u).sporulate());

            let kernels = kernels.join().unwrap();
            let sample_workspace = kernels.sample_workspace(compute).sporulate();

            Ok(Self(ManuallyDrop::new(Internal {
                embed_tokens,
                layers,
                lm_layernorm,
                lm_head,
                pool: Mutex::new(pool),

                config: host.config,
                resource: resource.clone(),
                transfer: transfer.sporulate(),

                kernels,
                sample_workspace,
            })))
        })
    }
}

impl Transformer {
    #[inline]
    fn cache(&self, len: usize) -> Cache {
        Cache::new(&self.0.resource, len)
    }

    #[inline]
    fn tensor(&self, shape: &[udim]) -> Tensor<Cache> {
        Tensor::alloc(self.0.config.dt, shape, |len| self.cache(len))
    }
}

impl CausalLM for Transformer {
    type Storage = Cache;

    #[inline]
    fn max_seq_len(&self) -> upos {
        self.0.config.max_seq_len
    }
    #[inline]
    fn bos_token(&self) -> utok {
        self.0.config.bos_token
    }
    #[inline]
    fn eos_token(&self) -> utok {
        self.0.config.eos_token
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        self.0.config.new_cache(|len| self.cache(len))
    }

    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        InferenceConfig::duplicate_cache(
            cache,
            pos,
            |len| self.cache(len),
            |dst, src| {
                self.0.resource.apply(|stream| {
                    let ctx = stream.ctx();
                    self.0.kernels.reform(
                        &mut dst.map_physical(|u| &mut **u.mem.sprout_mut(ctx)),
                        &src.map_physical(|u| &**u.mem.sprout_ref(ctx)),
                        stream,
                    );
                })
            },
        )
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;
        let d = self.0.config.d;

        let mut x = self.tensor(&[nt, d]);
        self.0.resource.apply(|compute| {
            self.0.kernels.gather(
                &mut x
                    .as_mut()
                    .map_physical(|u| &mut **u.mem.sprout_mut(compute.ctx())),
                &self.0.embed_tokens.as_ref().map_physical(|u| &**u),
                tokens,
                compute,
            )
        });
        x
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a,
    {
        self.0.resource.apply(|compute| {
            let ctx = compute.ctx();
            let transfer = self.0.transfer.sprout_ref(ctx);
            let stream = ComputeStream {
                nh: self.0.config.nh,
                nkvh: self.0.config.nkvh,
                di: self.0.config.di,
                epsilon: self.0.config.epsilon,
                theta: self.0.config.theta,
                kernels: &self.0.kernels,
                compute,
                transfer,
                host: &self.0.layers,
                dev: Rc::new(RefCell::new(self.0.pool.lock().unwrap())),
            };
            <ComputeStream as llama::ComputeStream>::forward(&stream, queries, token_embedded)
        })
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        mut hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        self.0.resource.apply(|compute| {
            let ctx = compute.ctx();
            let mut x = hidden_state
                .as_mut()
                .map_physical(|u| &mut **u.mem.sprout_mut(ctx));
            let range =
                DecodingMeta::select(&mut x, decoding, |dst, src| compute.memcpy_d2d(dst, src));
            if range.is_empty() {
                return self.tensor(&[0, self.0.config.d]);
            }

            let lm_layernorm = self
                .0
                .lm_layernorm
                .as_ref()
                .map_physical(|u| &**u.sprout_ref(ctx));
            let lm_head = self
                .0
                .lm_head
                .as_ref()
                .map_physical(|u| &**u.sprout_ref(ctx));

            let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
            let mut logits = self.tensor(&[x.shape()[0], lm_head.shape()[1]]);

            // 复制一个 x 以实现原地归一化
            let x_ = x
                .as_ref()
                .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
            self.0
                .kernels
                .rms_norm(&mut x, &x_, &lm_layernorm, self.0.config.epsilon, compute);
            self.0.kernels.mat_mul(
                &mut logits
                    .as_mut()
                    .map_physical(|u| &mut **u.mem.sprout_mut(ctx)),
                0.,
                &x,
                &lm_head,
                1.,
                compute,
            );

            logits
        })
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        let workspace_ptr = unsafe { self.0.sample_workspace.as_raw() };
        let workspace_len = self.0.sample_workspace.len();
        self.0.resource.apply(|compute| {
            let workspace =
                unsafe { from_raw_parts_mut(workspace_ptr as *mut DevByte, workspace_len) };
            self.0.kernels.sample(
                self.0.config.voc as _,
                args.into_iter()
                    .flat_map(|meta| repeat(meta.args).take(meta.num_decode)),
                logits.take_physical().mem.sprout_ref(compute.ctx()),
                workspace,
                compute,
            )
        })
    }
}

impl Drop for Transformer {
    #[inline]
    fn drop(&mut self) {
        let Internal {
            config: _,
            resource,
            transfer,
            kernels: _,
            sample_workspace,
            embed_tokens,
            layers,
            lm_layernorm,
            lm_head,
            pool,
        } = unsafe { ManuallyDrop::take(&mut self.0) };
        resource.apply(|compute| {
            let ctx = compute.ctx();
            transfer.sprout(ctx);
            sample_workspace.sprout(ctx);
            embed_tokens.take_physical().sprout(ctx);
            lm_layernorm.take_physical().sprout(ctx);
            lm_head.take_physical().sprout(ctx);
            for layer in layers {
                layer.att_layernorm.take_physical().sprout(ctx);
                layer.att_qkv.take_physical().sprout(ctx);
                layer.att_o.take_physical().sprout(ctx);
                layer.mlp_layernorm.take_physical().sprout(ctx);
                layer.mlp_gate_up.take_physical().sprout(ctx);
                layer.mlp_down.take_physical().sprout(ctx);
            }
            for (layer, event) in take(&mut *pool.lock().unwrap()) {
                layer.att_layernorm.take_physical().sprout(ctx);
                layer.att_qkv.take_physical().sprout(ctx);
                layer.att_o.take_physical().sprout(ctx);
                layer.mlp_layernorm.take_physical().sprout(ctx);
                layer.mlp_gate_up.take_physical().sprout(ctx);
                layer.mlp_down.take_physical().sprout(ctx);
                event.sprout(ctx);
            }
        });
    }
}

struct ComputeStream<'a> {
    nh: udim,
    nkvh: udim,
    di: udim,
    epsilon: f32,
    theta: f32,
    kernels: &'a NvidiaKernels,
    compute: &'a Stream<'a>,
    transfer: &'a Stream<'a>,
    host: &'a [LayerStorage<HostMemSpore>],
    dev: DevMemPool<'a>,
}

type DevMemPool<'a> =
    Rc<RefCell<MutexGuard<'a, VecDeque<(LayerStorage<DevMemSpore>, EventSpore)>>>>;

impl<'a> llama::ComputeStream for ComputeStream<'a> {
    type Handle = Gpu;
    type Storage = Cache;
    type Buf<'m> = DevMem<'m>;
    type Pos<'m> = DevMem<'m>;

    #[inline]
    fn malloc(&self, len: usize) -> Self::Buf<'_> {
        self.compute.malloc::<u8>(len)
    }
    #[inline]
    fn free(&self, mem: Self::Buf<'_>) {
        mem.drop_on(self.compute);
    }
    #[inline]
    fn map_pos<'b>(&self, pos: &'b [u32]) -> Self::Pos<'b>
    where
        Self: 'b,
    {
        self.compute.from_host(pos)
    }
    #[inline]
    fn free_pos(&self, mem: Self::Pos<'_>) {
        mem.drop_on(self.compute);
    }
    #[inline]
    fn map_storage<'b>(&'b self, storage: &'b mut Self::Storage) -> &'b mut SliceOn<Self::Handle> {
        storage.mem.sprout_mut(self.compute.ctx())
    }
    #[inline]
    fn kernels(&self) -> &impl Kernels<Self::Handle> {
        self.kernels
    }
    #[inline]
    fn queue(&self) -> &llama::QueueOf<Self::Handle> {
        self.compute
    }
    #[inline]
    fn constant(&self) -> ComputeConst {
        ComputeConst {
            nh: self.nh,
            nkvh: self.nkvh,
            di: self.di,
            epsilon: self.epsilon,
            theta: self.theta,
        }
    }

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = SliceOn<Self::Handle>>,
    {
        println!(
            "{}",
            tensor.as_ref().map_physical(|s| {
                let mut host = Blob::new(s.len());
                memcpy_d2h(&mut host, s);
                host
            })
        );
    }

    fn layers(
        &self,
    ) -> impl Iterator<Item = impl llama::LLamaLayer<Byte = <Self::Handle as llama::Handle>::Byte>>
    {
        Iter::new(self.host, self.dev.clone(), self.compute, self.transfer)
    }
}

struct Iter<'a> {
    host: &'a [LayerStorage<HostMemSpore>],
    pool: DevMemPool<'a>,
    compute: &'a Stream<'a>,
    transfer: &'a Stream<'a>,
    layer: usize,
}

impl<'a> Iter<'a> {
    pub fn new(
        host: &'a [LayerStorage<HostMemSpore>],
        pool: DevMemPool<'a>,
        compute: &'a Stream,
        transfer: &'a Stream,
    ) -> Self {
        Self {
            host,
            pool,
            compute,
            transfer,
            layer: 0,
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = LayerLoader<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.layer >= self.host.len() {
            return None;
        }

        let mut pool = self.pool.borrow_mut();
        let load = if pool.len() < self.host.len() {
            Some((self.layer + pool.len()) % self.host.len())
        } else {
            None
        };
        self.layer += 1;

        let (s, event) = pool.pop_front().unwrap();
        let ctx = self.compute.ctx();
        self.compute.wait_for(&event.sprout(ctx));

        Some(Self::Item {
            host: self.host,
            pool: self.pool.clone(),
            load,
            transfer: self.transfer,
            storage: Some(s),
        })
    }
}

struct LayerLoader<'a> {
    host: &'a [LayerStorage<HostMemSpore>],
    pool: DevMemPool<'a>,
    load: Option<usize>,
    transfer: &'a Stream<'a>,
    storage: Option<LayerStorage<DevMemSpore>>,
}

macro_rules! access {
    ($self:expr, $name:ident) => {
        $self
            .storage
            .as_ref()
            .unwrap()
            .$name
            .as_ref()
            .map_physical(|u| &**u.sprout_ref($self.transfer.ctx()))
    };
}
impl llama::LLamaLayer for LayerLoader<'_> {
    type Byte = DevByte;
    type Storage<'m>
        = &'m [DevByte]
    where
        Self: 'm;

    fn att_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, att_layernorm)
    }
    fn att_qkv(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, att_qkv)
    }
    fn att_o(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, att_o)
    }
    fn mlp_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, mlp_layernorm)
    }
    fn mlp_gate_up(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, mlp_gate_up)
    }
    fn mlp_down(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, mlp_down)
    }
}

impl Drop for LayerLoader<'_> {
    fn drop(&mut self) {
        let mut lll = self.storage.take().unwrap();
        if let Some(load) = self.load {
            macro_rules! exchange {
                ($($name:ident)+) => {
                    $(
                        let host = self.host[load].$name.physical();
                        let mut dev = lll.$name.physical_mut().sprout_mut(self.transfer.ctx());
                        self.transfer.memcpy_h2d(&mut dev, host);
                    )+
                };
            }
            exchange! {
                att_layernorm
                att_qkv
                att_o
                mlp_layernorm
                mlp_gate_up
                mlp_down
            }
        }
        self.pool
            .borrow_mut()
            .push_back((lll, self.transfer.record().sporulate()));
    }
}

#[test]
fn test_infer() {
    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }
    let device = cuda::Device::new(0);
    causal_lm::test_impl::<Transformer>(
        ModelLoadMeta {
            device,
            load_layers: 20,
        },
        &[
            29966, 29989, 1792, 29989, 29958, 13, 29903, 388, 376, 18567, 29908, 304, 592, 21106,
            29879, 5299, 29989, 465, 22137, 29989, 29958, 13,
        ],
    );
}
