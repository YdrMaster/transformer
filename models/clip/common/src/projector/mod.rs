pub(crate) mod resampler;

use gguf::{GGufMetaMapExt, GGufModel};

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
