pub mod args;
pub mod compute;
pub mod storage;

use gguf::ggml_quants::digit_layout::DigitLayout;
use std::ops::{Range, RangeBounds};

pub use args::{Args as LlamaArgs, Request as LlamaRequest};
pub use common::Contiguous;
pub use compute::{BlkWeight, Gpt2Worker, Operators, WeightLoader};
pub use storage::{BlkStorage, Storage};
pub use tensor::{RandomSample, Tensor};
pub mod ext {
    pub use gguf::{
        ext::{utok, Mmap},
        ggml_quants,
    };
}

#[derive(Clone, Debug)]
pub struct Gpt2Meta {
    pub dt_embd: DigitLayout,
    pub dt_token_embd: DigitLayout,   // 词汇编码布局
    pub dt_postion_embd: DigitLayout, // 位置编码布局
    pub dt_norm: DigitLayout,
    pub dt_mat: DigitLayout,

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

impl Gpt2Meta {
    pub fn distribute(&mut self, range: impl RangeBounds<usize>, count: usize) {
        let len = normalize(range, count).len();
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
            ..
        } = self;
        Tensor::new(dt_embd, &[buf, nblk, 2, nkvh, 64])
    }

    pub fn embd(&self, nt: usize) -> Tensor<usize> {
        let &Self { dt_embd, d, .. } = self;
        Tensor::new(dt_embd, &[nt, d])
    }

    pub fn logits(&self, nt: usize) -> Tensor<usize> {
        let &Self { dt_embd, nvoc, .. } = self;
        Tensor::new(dt_embd, &[nt, nvoc])
    }

    // wte
    pub fn token_embd(&self) -> Tensor<usize> {
        self.embd(self.nvoc)
    }
    // wpe
    pub fn position_embd(&self) -> Tensor<usize> {
        self.embd(self.nctx)
    }

    pub fn attn_qkv_w(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(3 * d, d, usage)
    }

    pub fn attn_qkv_b(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(3 * d, 1, usage)
    }

    pub fn attn_o_w(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, d, usage)
    }

    pub fn attn_o_b(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, 1, usage)
    }

    pub fn ffn_up_w(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(di, d, usage)
    }

    pub fn ffn_up_b(&self, _usage: TensorUsage) -> Tensor<usize> {
        Tensor::new(self.dt_embd, &[self.di])
    }

    pub fn ffn_down_w(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(d, di, usage)
    }

    pub fn ffn_down_b(&self, _usage: TensorUsage) -> Tensor<usize> {
        Tensor::new(self.dt_embd, &[self.d])
    }

    pub fn output_weight(&self) -> Tensor<usize> {
        Tensor::new(self.dt_embd, &[self.nvoc, self.d])
    }

    pub fn norm(&self) -> Tensor<usize> {
        let &Self { dt_norm, d, .. } = self;
        Tensor::new(dt_norm, &[d])
    }

    pub fn pos_embd(&self) -> Tensor<usize> {
        let &Self { nvoc, d, .. } = self;
        Tensor::new(self.dt_embd, &[nvoc, d])
    }

    fn mat(&self, row: usize, col: usize, usage: TensorUsage) -> Tensor<usize> {
        // NOTICE: 权重矩阵以 mat 类型存储但以 embd 类型参与计算
        match usage {
            TensorUsage::Storage => {
                Tensor::new(self.dt_mat, &[row, col / self.dt_mat.group_size()])
            }
            TensorUsage::Computation => {
                assert_eq!(self.dt_embd.group_size(), 1);
                Tensor::new(self.dt_embd, &[row, col]).transpose(&[1, 0])
            }
        }
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
