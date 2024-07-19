﻿#![cfg(detected_cuda)]

mod gather;

use ::sample::SampleArgs;
use common::{f16, utok};
use common_devices::{Operators, SliceOn};
use cuda::{AsRaw, Device};
use digit_layout::types::{F16, U32};
use operators::{
    cuda::{memcpy_d2h, DevByte, DevMem, Stream},
    dyn_,
    fuesd_softmax::nvidia_gpu as softmax,
    mat_mul::nvidia_gpu as mat_mul,
    random_sample::{nvidia_gpu as random_sample, KVPair, RandomSample},
    reform::nvidia_gpu as reform,
    rms_norm::nvidia_gpu as rms_norm,
    rope::nvidia_gpu as rope,
    swiglu::nvidia_gpu as swiglu,
    Operator, QueueOf, TensorLayout, Workspace,
};
use std::{
    collections::HashMap,
    mem::size_of,
    ops::{Deref, DerefMut},
    ptr::{null, null_mut},
};

pub use common_devices::{Kernels, KernelsA, KernelsB};
pub use operators::{cuda, nvidia_gpu::Handle as Gpu};
pub use tensor::{reslice, reslice_mut, slice, split, udim, LocalSplitable, Tensor};

#[cfg(detected_nccl)]
pub use operators::nccl;

pub struct NvidiaKernels(HashMap<i32, Internal>);

struct Internal {
    mat_mul: mat_mul::Operator,
    rms_norm: rms_norm::Operator,
    rope: rope::Operator,
    reform: reform::Operator,
    softmax: softmax::Operator,
    swiglu: swiglu::Operator,
    random_sample: random_sample::Operator,
}

impl Internal {
    pub fn new(handle: &Gpu, d: usize, voc: usize) -> Self {
        let mat_mul = mat_mul::Operator::new(handle);

        let mut rms_norm = rms_norm::Operator::new(handle);
        rms_norm
            .scheme(&operators::rms_norm::Args {
                y_layout: TensorLayout::new(F16, [dyn_(), d.into()], [dyn_(); 2]),
                y_base: null_mut(),
                x_layout: TensorLayout::new(F16, [dyn_(), d.into()], [dyn_(); 2]),
                x_base: null(),
                w_layout: TensorLayout::new(F16, [d.into()], [dyn_()]),
                w_base: null(),
                epsilon: 0.,
            })
            .unwrap();

        let mut rope = rope::Operator::new(handle);
        rope.scheme(&operators::rope::Args {
            t_layout: TensorLayout::new(F16, [dyn_(); 3], [dyn_(); 3]),
            t_base: null_mut(),
            p_layout: TensorLayout::new(U32, [dyn_()], [dyn_()]),
            p_base: null(),
            theta: 0.,
        })
        .unwrap();

        let mut reform = reform::Operator::new(handle);
        reform
            .scheme(&operators::reform::Args {
                dst_layout: TensorLayout::new(F16, [dyn_(); 2], [dyn_(); 2]),
                dst_base: null_mut(),
                src_layout: TensorLayout::new(F16, [dyn_(); 2], [dyn_(); 2]),
                src_base: null(),
            })
            .unwrap();

        let mut softmax = softmax::Operator::new(handle);
        softmax
            .scheme(&operators::fuesd_softmax::Args {
                att_layout: TensorLayout::new(F16, [dyn_(); 3], [dyn_(); 3]),
                att_base: null_mut(),
            })
            .unwrap();

        let mut swiglu = swiglu::Operator::new(handle);
        swiglu
            .scheme(&operators::swiglu::Args {
                gate_layout: TensorLayout::new(F16, [dyn_(); 2], [dyn_(); 2]),
                gate_base: null_mut(),
                up_layout: TensorLayout::new(F16, [dyn_(); 2], [dyn_(); 2]),
                up_base: null(),
            })
            .unwrap();

        let mut random_sample = random_sample::Operator::new(handle);
        random_sample
            .scheme(&operators::random_sample::Args::new(F16, voc))
            .unwrap();

        Self {
            mat_mul,
            rms_norm,
            rope,
            reform,
            softmax,
            swiglu,
            random_sample,
        }
    }
}

impl NvidiaKernels {
    pub fn new(devices: &[Device], rms_norm_size: usize, voc_size: usize) -> Self {
        Self(
            devices
                .iter()
                .map(|d| {
                    (
                        unsafe { d.as_raw() },
                        Internal::new(&Gpu::new(d.retain_primary()), rms_norm_size, voc_size),
                    )
                })
                .collect(),
        )
    }

    fn get(&self, queue: &QueueOf<Gpu>) -> &Internal {
        self.0.get(&unsafe { queue.ctx().dev().as_raw() }).unwrap()
    }

    pub fn sample_workspace<'ctx>(&self, queue: &'ctx QueueOf<Gpu>) -> DevMem<'ctx> {
        let random_sample = &self.get(queue).random_sample;
        let workspace_len = random_sample.workspace();
        let scheme_n = random_sample.scheme_n();
        let mut workspace = queue.malloc::<u8>(workspace_len);
        let host = (0..scheme_n).map(|i| i as u32).collect::<Vec<_>>();
        queue.memcpy_h2d(&mut workspace[..scheme_n * size_of::<u32>()], &host);
        workspace
    }

    pub fn sample(
        &self,
        args: impl IntoIterator<Item = SampleArgs>,
        logits: &[DevByte],
        workspace: &mut [DevByte],
        stream: &Stream,
    ) -> Vec<utok> {
        let random_sample = &self.get(stream).random_sample;
        let voc = random_sample.scheme_n();
        let logits = logits.as_ptr();

        let details = args.into_iter().collect::<Vec<_>>();
        let kv_pair_size = KVPair::<()>::LAYOUT.nbytes();
        let mut kv_pairs = stream.malloc::<u8>(details.len() * kv_pair_size);

        let mut args = operators::random_sample::Args::<Gpu>::new(F16, voc);
        args.workspace = Workspace {
            ptr: workspace.as_mut_ptr(),
            len: workspace.len(),
        };
        for (i, arg) in details.iter().enumerate() {
            args.kv_pair_base = unsafe { kv_pairs.as_mut_ptr().add(i * kv_pair_size) };
            args.data_base = unsafe { logits.add(i * voc * F16.nbytes()) };
            args.detail.temperature = arg.temperature;
            args.detail.top_p = arg.top_p;
            args.detail.top_k = arg.top_k;
            random_sample.launch(&args, stream).unwrap();
        }

        let mut host = vec![KVPair::new(0, f16::ZERO); details.len()];
        stream.synchronize();
        memcpy_d2h(&mut host, &kv_pairs);

        host.into_iter().map(|kv| kv.idx() as _).collect()
    }
}

impl Kernels<Gpu> for NvidiaKernels {}

impl Operators for NvidiaKernels {
    type Handle = Gpu;

    fn reform_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::reform::Reform<Self::Handle> {
        &self.get(queue).reform
    }

    fn rms_norm_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::rms_norm::RmsNorm<Self::Handle> {
        &self.get(queue).rms_norm
    }

    fn mat_mul_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::mat_mul::MatMul<Self::Handle> {
        &self.get(queue).mat_mul
    }

    fn rope_op(&self, queue: &QueueOf<Self::Handle>) -> &impl operators::rope::Rope<Self::Handle> {
        &self.get(queue).rope
    }

    fn softmax_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::fuesd_softmax::FusedSoftmax<Self::Handle> {
        &self.get(queue).softmax
    }

    fn swiglu_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::swiglu::Swiglu<Self::Handle> {
        &self.get(queue).swiglu
    }
}

impl KernelsB for NvidiaKernels {
    type Handle = Gpu;

    fn gather<T, U, I>(
        &self,
        x: &mut Tensor<T>,
        table: &Tensor<U>,
        tokens: I,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens, queue);
    }
}

pub fn synchronize() {
    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }
    for i in 0..cuda::Device::count() {
        cuda::Device::new(i as _)
            .retain_primary()
            .apply(|ctx| ctx.synchronize());
    }
}
