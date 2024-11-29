use crate::{ClipMeta, ProjectorType};
use gguf::{GGufMetaMapExt, GGufModel};

#[derive(Clone)]
pub struct Storage<T> {
    pub meta: ClipMeta,
    pub patch_embd_w: T,
    pub patch_embd_b: T,
    pub pos_embd: T,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        let pos_embd = &gguf.tensors["v.position_embd.weight"];
        let patch_embd_w = &gguf.tensors["v.patch_embd.weight"];
        let patch_embd_b = &gguf.tensors["v.patch_embd.bias"];

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

            dt_embd: pos_embd.ty,
            dt_mat :  patch_embd_w.ty,
            dt_bias:  patch_embd_b.ty,

            nblk   : gguf.get_usize("clip.vision.block_count"                 ).unwrap(),
            d_patch: gguf.get_usize("clip.vision.patch_size"                  ).unwrap(),
            d_image: gguf.get_usize("clip.vision.image_size"                  ).unwrap(),
            nh     : gguf.get_usize("clip.vision.attention.head_count"        ).unwrap(),
            d      : gguf.get_usize("clip.vision.embedding_length"            ).unwrap(),
            di     : gguf.get_usize("clip.vision.feed_forward_length"         ).unwrap(),

            image_mean: get_rgb(gguf, "clip.vision.image_mean"),
            image_std : get_rgb(gguf, "clip.vision.image_std" ),
            epsilon   : gguf.get_f32("clip.vision.attention.layer_norm_epsilon").unwrap(),
        };

        Self {
            meta,
            patch_embd_w: patch_embd_w.data,
            patch_embd_b: patch_embd_b.data,
            pos_embd: pos_embd.data,
        }
    }
}

fn get_rgb(gguf: &GGufModel, key: &str) -> [f32; 3] {
    let mut arr = gguf.get_f32_arr(key).unwrap();
    let mut ans = [0.0; 3];
    for x in ans.iter_mut() {
        *x = arr.next().unwrap().unwrap();
    }
    ans
}

#[test]
fn test() {
    use test_utils::Inference;
    let Some(Inference { model, .. }) = Inference::load() else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));
    let storage = Storage::from_gguf(&gguf);
    println!("{:#?}", storage.meta);
}
