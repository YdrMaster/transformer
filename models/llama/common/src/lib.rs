mod args;
mod compute;
mod storage;

use std::ops::{Range, RangeBounds};

pub use args::{Args as LlamaArgs, Request as LlamaRequest};
pub use common::Contiguous;
pub use compute::{BlkWeight, LlamaWorker, Operators, WeightLoader};
pub use storage::{BlkStorage as LlamaBlkStorage, Storage as LlamaStorage};
pub use tensor::{RandomSample, Tensor};
pub mod ext {
    pub use gguf::{
        ext::{utok, Mmap},
        ggml_quants,
        ggml_quants::{
            digit_layout::{types as primitive, DigitLayout},
            f16, types as quant,
        },
    };
}

#[derive(Clone, Debug)]
pub struct LlamaMeta {
    pub dt_embd: ext::DigitLayout,
    pub dt_norm: ext::DigitLayout,
    pub dt_mat: ext::DigitLayout,

    pub nblk: usize,
    pub nctx: usize,
    pub nvoc: usize,
    pub nh: usize,
    pub nkvh: usize,
    pub d: usize,
    pub dh: usize,
    pub di: usize,

    pub epsilon: f32,
    pub theta: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TensorUsage {
    Storage,
    Computation,
}

impl LlamaMeta {
    pub fn distribute(&mut self, len: usize, count: usize) {
        assert!(0 < len && len <= count);
        assert_eq!(self.nkvh % count, 0);
        assert_eq!(self.di % count, 0);

        self.nh = self.nh / count * len;
        self.nkvh = self.nkvh / count * len;
        self.di = self.di / count * len;
    }

    pub fn kv_cache(&self, buf: usize) -> Tensor<usize> {
        let &Self {
            dt_embd,
            nblk,
            nkvh,
            dh,
            ..
        } = self;
        Tensor::new(dt_embd, &[buf, nblk, 2, nkvh, dh])
    }

    pub fn embd(&self, nt: usize) -> Tensor<usize> {
        let &Self { dt_embd, d, .. } = self;
        Tensor::new(dt_embd, &[nt, d])
    }

    pub fn logits(&self, nt: usize) -> Tensor<usize> {
        let &Self { dt_embd, nvoc, .. } = self;
        Tensor::new(dt_embd, &[nt, nvoc])
    }

    pub fn token_embd(&self) -> Tensor<usize> {
        self.embd(self.nvoc)
    }

    pub fn attn_norm(&self) -> Tensor<usize> {
        self.norm()
    }

    pub fn attn_qkv(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self {
            nh, nkvh, d, dh, ..
        } = self;
        self.mat((nh + nkvh + nkvh) * dh, d, usage)
    }

    pub fn attn_o(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { nh, d, dh, .. } = self;
        self.mat(d, nh * dh, usage)
    }

    pub fn ffn_norm(&self) -> Tensor<usize> {
        self.norm()
    }

    pub fn ffn_gate_up(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(di + di, d, usage)
    }

    pub fn ffn_down(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(d, di, usage)
    }

    pub fn output_norm(&self) -> Tensor<usize> {
        self.norm()
    }

    pub fn output(&self) -> Tensor<usize> {
        self.token_embd().transpose(&[1, 0])
    }

    fn norm(&self) -> Tensor<usize> {
        let &Self { dt_norm, d, .. } = self;
        Tensor::new(dt_norm, &[d])
    }

    fn mat(&self, row: usize, col: usize, usage: TensorUsage) -> Tensor<usize> {
        Tensor::new(
            // NOTICE: 权重矩阵以 mat 类型存储但以 embd 类型参与计算
            match usage {
                TensorUsage::Storage => self.dt_mat,
                TensorUsage::Computation => self.dt_embd,
            },
            &[row, col],
        )
        .transpose(&[1, 0])
    }
}

fn normalize(range: impl RangeBounds<usize>, count: usize) -> Range<usize> {
    use std::ops::Bound::{Excluded, Included, Unbounded};
    let start = match range.start_bound() {
        Included(&i) => i,
        Excluded(&i) => i + 1,
        Unbounded => 0,
    };
    let end = match range.end_bound() {
        Included(&i) => i + 1,
        Excluded(&i) => i,
        Unbounded => count,
    };
    assert!(start < end && end <= count);
    start..end
}
