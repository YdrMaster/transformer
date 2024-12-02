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
    // ln1_weight
    pub fn attn_norm_weight(&self) -> Tensor<usize> {
        self.norm()
    }
    // ln1_bias
    pub fn attn_norm_bias(&self) -> Tensor<usize> {
        self.norm()
    }
    // attn_qkvw
    pub fn attn_qkv_weight(&self, usage: TensorUsage) -> Tensor<usize> {
        self.mat(3 * self.d, self.d, usage)
    }
    // attn_qkvb
    pub fn attn_qkv_bias(&self) -> Tensor<usize> {
        Tensor::new(self.dt_embd, &[3 * self.d])
    }
    // attn_projw
    pub fn attn_o_weight(&self, usage: TensorUsage) -> Tensor<usize> {
        self.mat(self.d, self.d, usage)
    }
    // attn_projb
    pub fn attn_o_bias(&self) -> Tensor<usize> {
        Tensor::new(self.dt_embd, &[self.d])
    }
    // ln2_weight
    pub fn ffn_norm_weight(&self) -> Tensor<usize> {
        self.norm()
    }
    // ln2_bias
    pub fn ffn_norm_bias(&self) -> Tensor<usize> {
        self.norm()
    }
    // fcw
    pub fn ffn_up_weight(&self, usage: TensorUsage) -> Tensor<usize> {
        self.mat(4 * self.d, self.d, usage)
    }
    // fcb
    pub fn ffn_up_bias(&self) -> Tensor<usize> {
        Tensor::new(self.dt_embd, &[4 * self.d])
    }
    // fcprojw
    pub fn ffn_down_weight(&self, usage: TensorUsage) -> Tensor<usize> {
        self.mat(self.d, 4 * self.d, usage)
    }
    // fcprojb
    pub fn ffn_down_bias(&self) -> Tensor<usize> {
        Tensor::new(self.dt_embd, &[self.d])
    }
    // lnfw
    pub fn output_norm_weight(&self) -> Tensor<usize> {
        self.norm()
    }
    // lnfb
    pub fn output_norm_bias(&self) -> Tensor<usize> {
        self.norm()
    }
    // output.weight
    pub fn output_weight(&self) -> Tensor<usize> {
        Tensor::new(self.dt_embd, &[self.nvoc, self.d])
    }

    fn norm(&self) -> Tensor<usize> {
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
