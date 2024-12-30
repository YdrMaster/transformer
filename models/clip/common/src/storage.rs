use crate::{ClipMeta, ProjectorType};
use gguf::{meta, GGufMetaMapExt, GGufModel};

#[derive(Clone)]
pub struct Storage<T> {
    pub meta: ClipMeta,
    pub patch_embd_w: T,
    pub patch_embd_b: T,
    pub pos_embd: T,
    pub pre_norm: Option<[T; 2]>,
    pub post_norm: Option<[T; 2]>,
    pub blocks: Box<[BlkStorage<T>]>,
}

#[derive(Clone, Copy)]
pub struct BlkStorage<T> {
    pub attn_norm_w: T,
    pub attn_norm_b: T,
    pub attn_qkv_w: T,
    pub attn_qkv_b: T,
    pub attn_o_w: T,
    pub attn_o_b: T,

    pub ffn_norm_w: T,
    pub ffn_norm_b: T,
    pub ffn_up_w: T,
    pub ffn_up_b: T,
    pub ffn_down_w: T,
    pub ffn_down_b: T,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        let pos_embd = &gguf.tensors["v.position_embd.weight"];

        let projector = match gguf.get_str("clip.projector_type").unwrap() {
            "mlp" => ProjectorType::Mlp,
            "ldp" => ProjectorType::Ldp,
            "ldpv2" => ProjectorType::LdpV2,
            "resampler" => ProjectorType::Resampler,
            _ => ProjectorType::Unknown,
        };

        let d = meta![gguf => (usize) "clip.vision.embedding_length"];
        let nh = meta![gguf => (usize) "clip.vision.attention.head_count"];

        #[rustfmt::skip]
        let meta = ClipMeta {
            projector,
            minicpmv_version: meta![gguf => (usize) "clip.minicpmv_version"] as _,

            dt     : pos_embd.ty,
            d_patch: meta![gguf => (usize) "clip.vision.patch_size"],
            d_image: meta![gguf => (usize) "clip.vision.image_size"],

            d, nh,
            nblk: meta![gguf => (usize) "clip.vision.block_count"                 ],
            nkvh: meta![gguf => (usize) "clip.vision.attention.head_count_kv";  nh],
            dh  : meta![gguf => (usize) "clip.vision.rope_dimension_count"; d / nh],
            di  : meta![gguf => (usize) "clip.vision.feed_forward_length"         ],

            image_mean: get_rgb(gguf, "clip.vision.image_mean"),
            image_std : get_rgb(gguf, "clip.vision.image_std" ),
            epsilon   : gguf.get_f32("clip.vision.attention.layer_norm_epsilon").unwrap(),
        };
        #[rustfmt::skip]
        let blocks = (0..meta.nblk)
            .map(|i| BlkStorage {
                attn_norm_w: gguf.tensors[&*format!("v.blk.{i}.ln1.weight"     )].data,
                attn_norm_b: gguf.tensors[&*format!("v.blk.{i}.ln1.bias"       )].data,
                attn_qkv_w:  gguf.tensors[&*format!("v.blk.{i}.attn_qkv.weight")].data,
                attn_qkv_b:  gguf.tensors[&*format!("v.blk.{i}.attn_qkv.bias"  )].data,
                attn_o_w:    gguf.tensors[&*format!("v.blk.{i}.attn_out.weight")].data,
                attn_o_b:    gguf.tensors[&*format!("v.blk.{i}.attn_out.bias"  )].data,

                ffn_norm_w:  gguf.tensors[&*format!("v.blk.{i}.ln2.weight"     )].data,
                ffn_norm_b:  gguf.tensors[&*format!("v.blk.{i}.ln2.bias"       )].data,
                ffn_up_w:    gguf.tensors[&*format!("v.blk.{i}.ffn_up.weight"  )].data,
                ffn_up_b:    gguf.tensors[&*format!("v.blk.{i}.ffn_up.bias"    )].data,
                ffn_down_w:  gguf.tensors[&*format!("v.blk.{i}.ffn_down.weight")].data,
                ffn_down_b:  gguf.tensors[&*format!("v.blk.{i}.ffn_down.bias"  )].data,
            })
            .collect();

        Self {
            meta,
            patch_embd_w: gguf.tensors["v.patch_embd.weight"].data,
            patch_embd_b: gguf.tensors["v.patch_embd.bias"].data,
            pos_embd: pos_embd.data,
            pre_norm: gguf
                .tensors
                .get("v.pre_ln.weight")
                .map(|w| [w.data, gguf.tensors["v.pre_ln.bias"].data]),
            post_norm: gguf
                .tensors
                .get("v.post_ln.weight")
                .map(|w| [w.data, gguf.tensors["v.post_ln.bias"].data]),
            blocks,
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
