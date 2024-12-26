use crate::Gpt2Meta;
use gguf::{ext::Mmap, map_files, GGufMetaMapExt, GGufModel};
use std::path::Path;

#[derive(Clone)]
pub struct Storage<T> {
    pub meta: Gpt2Meta,
    pub token_embd: T,
    pub pos_embd: T,
    pub blocks: Box<[BlkStorage<T>]>,
    pub output_norm_b: T,
    pub output_norm_w: T,
    pub output: T,
}

#[derive(Clone, Copy)]
pub struct BlkStorage<T> {
    pub attn_qkv_b: T,
    pub attn_qkv_w: T,
    pub attn_o_b: T,
    pub attn_o_w: T,
    pub attn_norm_b: T,
    pub attn_norm_w: T,

    pub ffn_up_b: T,
    pub ffn_up_w: T,
    pub ffn_down_b: T,
    pub ffn_down_w: T,
    pub ffn_norm_b: T,
    pub ffn_norm_w: T,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        let token_embd = &gguf.tensors["token_embd.weight"];
        let position_embd = &gguf.tensors["position_embd.weight"];
        let output_norm_b = &gguf.tensors["output_norm.bias"];
        let output_norm_w = &gguf.tensors["output_norm.weight"];
        let output = &gguf.tensors["output.weight"];
        let qkv0 = &gguf.tensors["blk.0.attn_qkv.weight"];
        #[rustfmt::skip]
        let meta = Gpt2Meta {
            dt_embd:  token_embd.ty,
            dt_token_embd:  token_embd.ty,
            dt_postion_embd: position_embd.ty,
            dt_norm: output_norm_w.ty,
            dt_mat :               qkv0.ty,

            nblk: gguf.llm_block_count            ().unwrap(),
            nctx: gguf.llm_context_length         ().unwrap(),
            nvoc: gguf.tokenizer_ggml_tokens      ().unwrap().len(),
            nh  : gguf.llm_attention_head_count   ().unwrap(),
            nkvh: gguf.llm_attention_head_count_kv().unwrap(),
            d   : gguf.llm_embedding_length       ().unwrap(),
            dh  : gguf.llm_embedding_length       ().unwrap()/gguf.llm_attention_head_count   ().unwrap(),
            di  : gguf.llm_feed_forward_length    ().unwrap(),
            epsilon: 1e-5,
            theta: 1e4,
        };
        #[rustfmt::skip]
        let blocks = (0..meta.nblk)
            .map(|i| BlkStorage {
                attn_norm_w: gguf.tensors[&*format!("blk.{i}.attn_norm.weight"  )].data,
                attn_norm_b: gguf.tensors[&*format!("blk.{i}.attn_norm.bias"    )].data,
                attn_qkv_w:  gguf.tensors[&*format!("blk.{i}.attn_qkv.weight"   )].data,
                attn_qkv_b:  gguf.tensors[&*format!("blk.{i}.attn_qkv.bias"     )].data,
                attn_o_w:    gguf.tensors[&*format!("blk.{i}.attn_output.weight")].data,
                attn_o_b:    gguf.tensors[&*format!("blk.{i}.attn_output.bias"  )].data,

                ffn_norm_w:  gguf.tensors[&*format!("blk.{i}.ffn_norm.weight"   )].data,
                ffn_norm_b:  gguf.tensors[&*format!("blk.{i}.ffn_norm.bias"     )].data,
                ffn_up_w:    gguf.tensors[&*format!("blk.{i}.ffn_up.weight"     )].data,
                ffn_up_b:    gguf.tensors[&*format!("blk.{i}.ffn_up.bias"       )].data,
                ffn_down_w:  gguf.tensors[&*format!("blk.{i}.ffn_down.weight"   )].data,
                ffn_down_b:  gguf.tensors[&*format!("blk.{i}.ffn_down.bias"     )].data,
            })
            .collect();

        Self {
            meta,
            token_embd: token_embd.data,
            pos_embd: position_embd.data,
            blocks,
            output_norm_b: output_norm_b.data,
            output_norm_w: output_norm_w.data,
            output: output.data,
        }
    }
}

impl<T> BlkStorage<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> BlkStorage<U> {
        BlkStorage {
            attn_norm_b: f(self.attn_norm_b),
            attn_norm_w: f(self.attn_norm_w),
            attn_qkv_b: f(self.attn_qkv_b),
            attn_qkv_w: f(self.attn_qkv_w),
            attn_o_b: f(self.attn_o_b),
            attn_o_w: f(self.attn_o_w),

            ffn_up_b: f(self.ffn_up_b),
            ffn_up_w: f(self.ffn_up_w),
            ffn_down_b: f(self.ffn_down_b),
            ffn_down_w: f(self.ffn_down_w),
            ffn_norm_b: f(self.ffn_norm_b),
            ffn_norm_w: f(self.ffn_norm_w),
        }
    }

    pub fn as_ref(&self) -> BlkStorage<&T> {
        BlkStorage {
            attn_norm_b: &self.attn_norm_b,
            attn_norm_w: &self.attn_norm_w,
            attn_qkv_b: &self.attn_qkv_b,
            attn_qkv_w: &self.attn_qkv_w,
            attn_o_b: &self.attn_o_b,
            attn_o_w: &self.attn_o_w,

            ffn_up_b: &self.ffn_up_b,
            ffn_up_w: &self.ffn_up_w,
            ffn_down_b: &self.ffn_down_b,
            ffn_down_w: &self.ffn_down_w,
            ffn_norm_b: &self.ffn_norm_b,
            ffn_norm_w: &self.ffn_norm_w,
        }
    }
}

pub fn map_gguf_files() -> Option<Box<[Mmap]>> {
    let Some(path) = std::env::var_os("TEST_MODEL") else {
        println!("TEST_MODEL not set");
        return None;
    };
    let path = Path::new(&path);
    if !path.is_file() {
        println!("{path:?} not found");
        return None;
    }
    Some(map_files(path))
}

#[test]
fn test_load() {
    let Some(shards) = map_gguf_files() else {
        return;
    };
    let gguf = GGufModel::read(shards.iter().map(|s| &**s));
    let gpt2 = Storage::from_gguf(&gguf);
    println!("{:?}", gpt2.meta);
}
