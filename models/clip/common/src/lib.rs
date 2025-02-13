mod args;
mod compute;
mod image;
mod projector;
mod storage;

use gguf::ggml_quants::digit_layout::DigitLayout;
use projector::ProjectorMeta;

pub use args::Args as ClipArgs;
pub use compute::{BlkWeight, ClipWorker, Operators, WeightLoader};
pub use image::{Image, ImageGrid};
pub use projector::ProjectorStroage;
pub use storage::{BlkStorage as ClipBlkStorage, Storage as ClipStorage};
pub use tensor::Tensor;
pub mod ext {
    pub use gguf::{
        ext::{utok, Mmap},
        ggml_quants,
    };
}

#[derive(Clone, Debug)]
pub struct ClipMeta {
    pub dt: DigitLayout,

    pub d_patch: usize,
    pub d_image: usize,

    pub nblk: usize,
    pub nh: usize,
    pub nkvh: usize,
    pub d: usize,
    pub dh: usize,
    pub di: usize,

    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub epsilon: f32,

    pub projector: ProjectorMeta,
}

pub const D_POS_EMBD: usize = 70;

impl ClipMeta {
    pub fn embd(&self, np: usize) -> Tensor<usize> {
        let &Self { dt, d, .. } = self;
        Tensor::new(dt, &[np, d])
    }

    pub fn pos_embd(&self) -> Tensor<usize> {
        let &Self { dt, d, .. } = self;
        Tensor::new(dt, &[D_POS_EMBD.pow(2), d])
    }

    pub fn patch_embd_w(&self) -> Tensor<usize> {
        let &Self { d, d_patch, .. } = self;
        Tensor::new(self.dt, &[d, 3, d_patch, d_patch])
    }

    pub fn patch_embd_b(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        Tensor::new(self.dt, &[d])
    }

    pub fn norm(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        Tensor::new(self.dt, &[d])
    }

    pub fn attn_qkv_w(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(3 * d, d)
    }

    pub fn attn_qkv_b(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(3 * d, 1)
    }

    pub fn attn_o_w(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, d)
    }

    pub fn attn_o_b(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, 1)
    }

    pub fn ffn_up_w(&self) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(di, d)
    }

    pub fn ffn_up_b(&self) -> Tensor<usize> {
        let &Self { di, .. } = self;
        self.mat(di, 1)
    }

    pub fn ffn_down_w(&self) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(d, di)
    }

    pub fn ffn_down_b(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, 1)
    }

    fn mat(&self, row: usize, col: usize) -> Tensor<usize> {
        assert_eq!(self.dt.group_size(), 1);
        Tensor::new(self.dt, &[row, col]).transpose(&[1, 0])
    }
}
