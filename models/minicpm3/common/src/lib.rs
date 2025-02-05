mod args;
mod compute;
mod storage;

use common::Distribution;
use gguf::ggml_quants::digit_layout::DigitLayout;

pub use args::{Args as MiniCPM3Args, Request as MiniCPM3Request};
pub use compute::{Minicpm3Worker, Operators, WeightLoader};
pub use storage::{BlkStorage as MiniCPM3BlkStorage, Storage as MiniCPM3Storage};
pub use tensor::{RandomSample, Tensor};
pub mod ext {
    pub use gguf::{
        ext::{utok, Mmap},
        ggml_quants,
    };
}

#[derive(Clone, Debug)]
pub struct MiniCPM3Meta {
    pub dt_embd: DigitLayout,
    pub dt_norm: DigitLayout,
    pub dt_linear: DigitLayout,

    pub nblk: usize,
    pub nctx: usize,
    pub nvoc: usize,
    pub nh: usize,
    pub nkvh: usize,
    pub d: usize,
    pub dh: usize,
    pub di: usize,

    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub key_length: usize,

    pub epsilon: f32,
    pub theta: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TensorUsage {
    Storage,
    Computation,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MiniCPM3BlkWeight {
    AttnNorm,
    AttnO,
    FfnNorm,
    FfnGateUp,
    FfnDown,
}

impl MiniCPM3Meta {
    pub fn distribute(&self, dist: Distribution) -> Self {
        let [_, len, total] = dist.info();
        assert_eq!(self.nkvh % total, 0);
        assert_eq!(self.di % total, 0);

        Self {
            nh: self.nh / total * len,
            nkvh: self.nkvh / total * len,
            di: self.di / total * len,
            ..self.clone()
        }
    }

    pub fn blk(&self) -> MiniCPM3BlkStorage<usize> {
        use TensorUsage::Storage as TensorMem;
        let norm = self.norm().take();
        MiniCPM3BlkStorage {
            attn_norm: norm,
            attn_o: self.attn_o(TensorMem).take(),
            ffn_norm: norm,
            ffn_gate_up: self.ffn_gate_up(TensorMem).take(),
            ffn_down: self.ffn_down(TensorMem).take(),
        }
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

    pub fn norm(&self) -> Tensor<usize> {
        let &Self { dt_norm, d, .. } = self;
        Tensor::new(dt_norm, &[d])
    }

    pub fn token_embd(&self) -> Tensor<usize> {
        self.embd(self.nvoc)
    }

    pub fn attn_o(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { nh, d, dh, .. } = self;
        self.mat(d, nh * dh, usage)
    }

    pub fn ffn_gate_up(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(di + di, d, usage)
    }

    pub fn ffn_down(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(d, di, usage)
    }

    pub fn output(&self) -> Tensor<usize> {
        self.token_embd().transpose(&[1, 0])
    }

    fn mat(&self, row: usize, col: usize, usage: TensorUsage) -> Tensor<usize> {
        let &Self {
            dt_embd, dt_linear, ..
        } = self;
        // NOTICE: 权重矩阵以 mat 类型存储但以 embd 类型参与计算
        match usage {
            TensorUsage::Storage => Tensor::new(dt_linear, &[row, col / dt_linear.group_size()]),
            TensorUsage::Computation => {
                assert_eq!(dt_embd.group_size(), 1);
                Tensor::new(dt_embd, &[row, col]).transpose(&[1, 0])
            }
        }
    }
}
