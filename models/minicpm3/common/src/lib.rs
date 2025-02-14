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

    pub dq_lora: usize,
    pub dkv_lora: usize,
    pub dk: usize,
    pub dv: usize,
    pub dnope: usize,

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
    AttnQB,
    AttnQA,
    AttnKvB,
    AttnKvA,
    AttnQANorm,
    AttnKvANorm,
    AttnO,
    FfnNorm,
    FfnGateUp,
    FfnGate,
    FfnUp,
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
            attn_qa: self.attn_qa(TensorMem).take(),
            attn_qb: self.attn_qb(TensorMem).take(),
            attn_kvb: self.attn_kvb(TensorMem).take(),
            attn_kva: self.attn_kva(TensorMem).take(),
            attn_qa_norm: self.attn_qa_norm().take(),
            attn_kva_norm: self.attn_kva_norm().take(),
            attn_o: self.attn_o(TensorMem).take(),
            ffn_norm: norm,
            ffn_gate_up: self.ffn_gate_up(TensorMem).take(),
            ffn_down: self.ffn_down(TensorMem).take(),
            ffn_gate: self.ffn(TensorMem).take(),
            ffn_up: self.ffn(TensorMem).take(),
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
    pub fn token_embd(&self) -> Tensor<usize> {
        self.embd(self.nvoc)
    }
    pub fn norm(&self) -> Tensor<usize> {
        let &Self { dt_norm, d, .. } = self;
        Tensor::new(dt_norm, &[d])
    }
    // TODO  未实现分布
    pub fn attn_qa(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self {
            dt_embd,
            dq_lora,
            d,
            ..
        } = self;
        Tensor::new(dt_embd, &[dq_lora, d])
    }
    pub fn attn_qa_norm(&self) -> Tensor<usize> {
        let &Self {
            dt_norm, dq_lora, ..
        } = self;
        Tensor::new(dt_norm, &[dq_lora])
    }
    // TODO  未实现分布
    pub fn attn_qb(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self {
            dt_embd,
            nh,
            dk,
            dq_lora,
            ..
        } = self;
        Tensor::new(dt_embd, &[nh * dk, dq_lora])
    }

    // TODO  为实现分布式
    pub fn attn_kvb(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self {
            dt_embd,
            nh,
            dkv_lora,
            dnope,
            dv,
            ..
        } = self;
        Tensor::new(dt_embd, &[nh * (dnope + dv), dkv_lora])
    }
    // TODO  为实现分布式
    pub fn attn_kva(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self {
            dt_embd,
            dkv_lora,
            dh,
            d,
            ..
        } = self;
        Tensor::new(dt_embd, &[dkv_lora + dh, d])
    }
    pub fn attn_kva_norm(&self) -> Tensor<usize> {
        let &Self {
            dt_norm, dkv_lora, ..
        } = self;
        Tensor::new(dt_norm, &[dkv_lora])
    }
    pub fn attn_o(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { nh, d, dh, .. } = self;
        self.mat(d, d, usage)
    }

    pub fn ffn_gate_up(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(di + di, d, usage)
    }

    pub fn ffn_down(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(d, di, usage)
    }
    pub fn ffn(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(di, d, usage)
    }
    // TODO
    pub fn factor(&self) -> [Tensor<usize>; 2] {
        let &Self { dt_norm, dh, .. } = self;
        [
            Tensor::new(dt_norm, &[dh / 2]),
            Tensor::new(dt_norm, &[dh / 2]),
        ]
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
