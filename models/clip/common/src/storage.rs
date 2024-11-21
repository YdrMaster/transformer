use crate::{ClipMeta, ProjectorType};
use gguf::{GGufMetaMapExt, GGufModel};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Storage<T> {
    pub meta: ClipMeta,
    _phantom: PhantomData<T>,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        let position_embd = &gguf.tensors["v.position_embd.weight"];
        let w_patch_embd = &gguf.tensors["v.patch_embd.weight"];
        let b_patch_embd = &gguf.tensors["v.patch_embd.bias"];

        let projector = match gguf.get_str("clip.projector_type").unwrap() {
            "mlp" => ProjectorType::Mlp,
            "ldp" => ProjectorType::Ldp,
            "ldpv2" => ProjectorType::LdpV2,
            "resampler" => ProjectorType::Resampler,
            _ => ProjectorType::Unknown,
        };

        #[rustfmt::skip]
        let meta = ClipMeta {
            projector,
            minicpmv_version: gguf.get_usize("clip.minicpmv_version").unwrap() as _,

            dt_embd: position_embd.ty,
            dt_mat :  w_patch_embd.ty,
            dt_bias:  b_patch_embd.ty,

            nblk   : gguf.get_usize("clip.vision.block_count"                 ).unwrap(),
            d_patch: gguf.get_usize("clip.vision.patch_size"                  ).unwrap(),
            d_image: gguf.get_usize("clip.vision.image_size"                  ).unwrap(),
            nh     : gguf.get_usize("clip.vision.attention.head_count"        ).unwrap(),
            d      : gguf.get_usize("clip.vision.embedding_length"            ).unwrap(),
            di     : gguf.get_usize("clip.vision.feed_forward_length"         ).unwrap(),
            epsilon: gguf.get_f32  ("clip.vision.attention.layer_norm_epsilon").unwrap(),
        };

        Self {
            meta,
            _phantom: PhantomData,
        }
    }
}
