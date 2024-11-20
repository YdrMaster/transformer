mod image;

use gguf::ggml_quants::digit_layout::DigitLayout;
use tensor::Tensor;

pub use image::{Image, ImageGrid};

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

    pub epsilon: f32,
}

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

    pub fn embd(&self) -> Tensor<usize> {
        Tensor::new(self.dt_embd, &[self.n_patch(), self.n_mmproj_embd()])
    }
}
