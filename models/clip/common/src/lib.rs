mod args;
mod compute;
mod image;
mod storage;

use gguf::ggml_quants::digit_layout::DigitLayout;

pub use args::Args as ClipArgs;
pub use compute::{ClipWorker, Operators, WeightLoader};
pub use image::{Image, ImageGrid};
pub use storage::Storage as ClipStorage;
pub use tensor::Tensor;
pub mod ext {
    pub use gguf::{
        ext::{utok, Mmap},
        ggml_quants,
    };
}

#[derive(Clone, Debug)]
pub struct ClipMeta {
    pub projector: ProjectorType,
    pub minicpmv_version: u8,

    pub dt_embd: DigitLayout,
    pub dt_mat: DigitLayout,
    pub dt_bias: DigitLayout,

    pub nblk: usize,
    pub d_patch: usize,
    pub d_image: usize,
    pub nh: usize,
    pub d: usize,
    pub di: usize,

    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub epsilon: f32,
}

pub const D_POS_EMBD: usize = 70;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ProjectorType {
    Mlp,
    MlpNorm,
    Ldp,
    LdpV2,
    Resampler,
    Unknown,
}

impl ClipMeta {
    pub fn n_patch(&self) -> usize {
        let &Self {
            d_image, d_patch, ..
        } = self;
        let n_patch = (d_image / d_patch).pow(2);
        match self.projector {
            ProjectorType::Resampler => match self.minicpmv_version {
                2 => 96,
                3 => 64,
                _ => n_patch,
            },
            ProjectorType::Ldp | ProjectorType::LdpV2 => n_patch / 4,
            _ => n_patch,
        }
    }

    pub fn n_mmproj_embd(&self) -> usize {
        match self.projector {
            ProjectorType::Resampler => match self.minicpmv_version {
                2 => 4096,
                3 => 3584,
                _ => unreachable!(),
            },
            _ => todo!(),
        }
    }

    pub fn patch_embd_w(&self) -> Tensor<usize> {
        let &Self { d, d_patch, .. } = self;
        Tensor::new(self.dt_mat, &[d, 3, d_patch, d_patch])
    }

    pub fn patch_embd_b(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        Tensor::new(self.dt_bias, &[d])
    }

    pub fn pos_embd(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        Tensor::new(self.dt_embd, &[D_POS_EMBD.pow(2), d])
    }
}
