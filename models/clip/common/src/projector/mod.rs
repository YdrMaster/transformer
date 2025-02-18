pub(crate) mod resampler;

use gguf::{ggml_quants::digit_layout::DigitLayout, GGufMetaMapExt, GGufModel};
use tensor::Tensor;

#[derive(Clone, Debug)]
pub enum ProjectorMeta {
    Resampler(resampler::Meta),
}

impl ProjectorMeta {
    pub fn from_gguf(gguf: &GGufModel) -> Self {
        match gguf.get_str("clip.projector_type").unwrap() {
            "resampler" => ProjectorMeta::Resampler(resampler::Meta::from_gguf(gguf)),
            projector => todo!("unsupported projector type: {projector}"),
        }
    }

    pub fn img_embd(&self, dt: DigitLayout, batch: usize) -> Tensor<usize> {
        match self {
            ProjectorMeta::Resampler(meta) => meta.img_embd(dt, batch),
        }
    }
}

#[derive(Clone)]
pub enum ProjectorStroage<T> {
    Resampler(resampler::Storage<T>),
}

impl<'a> ProjectorStroage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        match gguf.get_str("clip.projector_type").unwrap() {
            "resampler" => ProjectorStroage::Resampler(resampler::Storage::from_gguf(gguf)),
            projector => todo!("unsupported projector type: {projector}"),
        }
    }
}
