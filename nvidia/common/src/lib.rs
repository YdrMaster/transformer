﻿#![cfg(detected_cuda)]

#[macro_use]
extern crate log;

mod fused_softmax;
mod gather;
mod mat_mul;
mod reform;
mod rms_norm;
mod rotary_embedding;
mod storage;
mod swiglu;

pub use common::utok;
pub use storage::{Cache, Storage};
pub use tensor::{slice, udim, DataType, Tensor};

use cublas::{Cublas, CublasSpore};
use cuda::{ContextGuard, ContextResource, ContextSpore, CudaDataType::half, DevSlice, Stream};
use fused_softmax::FusedSoftmax;
use reform::Reform;
use rms_norm::RmsNormalization;
use rotary_embedding::RotaryEmbedding;
use std::ops::{Deref, DerefMut};
use swiglu::Swiglu;
use transformer::{Kernels, Llama2};

pub struct NvidiaKernels {
    epsilon: f32,
    rms_norm: RmsNormalization,
    cublas: CublasSpore,
    theta: f32,
    rotary_embedding: RotaryEmbedding,
    reform: Reform,
    softmax: FusedSoftmax,
    swiglu: Swiglu,
}

impl NvidiaKernels {
    pub fn new(host: &dyn Llama2, ctx: &ContextGuard) -> Self {
        let dev = ctx.dev();
        let (block_size, _) = dev.max_block_dims();
        Self {
            epsilon: host.rms_norm_eps(),
            rms_norm: RmsNormalization::new(half, host.hidden_size(), block_size, ctx),
            cublas: Cublas::new(ctx).sporulate(),
            theta: host.rope_theta(),
            rotary_embedding: RotaryEmbedding::new(block_size, ctx),
            reform: Reform::new(block_size, 32, ctx),
            softmax: FusedSoftmax::new(half, host.max_position_embeddings(), block_size, ctx),
            swiglu: Swiglu::new(half, block_size, ctx),
        }
    }

    pub fn kill(&mut self, ctx: &ContextGuard) {
        self.rms_norm.kill(ctx);
        self.rotary_embedding.kill(ctx);
        self.reform.kill(ctx);
        self.softmax.kill(ctx);
        self.swiglu.kill(ctx);
    }
}

pub struct KernelRuntime<'a> {
    pub kernels: &'a NvidiaKernels,
    pub stream: &'a Stream<'a>,
}

impl NvidiaKernels {
    #[inline]
    pub fn on<'a>(&'a self, stream: &'a Stream) -> KernelRuntime<'a> {
        unsafe { self.cublas.sprout(stream.ctx()) }.set_stream(stream);
        KernelRuntime {
            kernels: self,
            stream,
        }
    }
}

impl Kernels for KernelRuntime<'_> {
    type Storage = DevSlice;

    #[inline]
    fn gather<T, U, I>(&self, x: &mut Tensor<T>, table: &Tensor<U>, tokens: I)
    where
        T: DerefMut<Target = DevSlice>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens, self.stream);
    }

    #[inline]
    fn rms_norm<T, U, V>(&self, y: &mut Tensor<T>, x: &Tensor<U>, w: &Tensor<V>)
    where
        T: DerefMut<Target = DevSlice>,
        U: Deref<Target = DevSlice>,
        V: Deref<Target = DevSlice>,
    {
        self.kernels
            .rms_norm
            .launch(y, x, w, self.kernels.epsilon, self.stream);
    }

    #[inline]
    fn mat_mul<T, U, V>(
        &self,
        c: &mut Tensor<T>,
        beta: f32,
        a: &Tensor<U>,
        b: &Tensor<V>,
        alpha: f32,
    ) where
        T: DerefMut<Target = DevSlice>,
        U: Deref<Target = DevSlice>,
        V: Deref<Target = DevSlice>,
    {
        let cublas = unsafe { self.kernels.cublas.sprout(self.stream.ctx()) };
        mat_mul::mat_mul(&cublas, c, beta, a, b, alpha)
    }

    #[inline]
    fn rotary_embedding<T, U>(&self, t: &mut Tensor<T>, pos: &Tensor<U>)
    where
        T: DerefMut<Target = DevSlice>,
        U: Deref<Target = DevSlice>,
    {
        self.kernels
            .rotary_embedding
            .launch(t, pos, self.kernels.theta, self.stream);
    }

    #[inline]
    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>)
    where
        T: DerefMut<Target = DevSlice>,
        U: Deref<Target = DevSlice>,
    {
        self.kernels.reform.launch(dst, src, self.stream);
    }

    #[inline]
    fn softmax<T>(&self, att: &mut Tensor<T>)
    where
        T: DerefMut<Target = DevSlice>,
    {
        self.kernels.softmax.launch(att, self.stream);
    }

    #[inline]
    fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>)
    where
        T: DerefMut<Target = DevSlice>,
        U: Deref<Target = DevSlice>,
    {
        self.kernels.swiglu.launch(gate, up, self.stream);
    }
}