use crate::{normalize, Gpt2Meta};
use common::{borrow, own, Contiguous};
use gguf::{GGufMetaMapExt, GGufModel};
use std::ops::{DerefMut, RangeBounds};
use tensor::Tensor;

use ext::Mmap;
use ggml_quants::digit_layout::DigitLayout;
use gguf::*;
use std::path::Path;

#[derive(Clone)]
pub struct Storage<T> {
    pub meta: Gpt2Meta,
    pub token_embd: T,
    pub pos_embd: T,
    pub blocks: Box<[BlkStorage<T>]>,
    pub output_norm_bias: T,
    pub output_norm_weight: T,
    pub output: T,
}

#[derive(Clone, Copy)]
pub struct BlkStorage<T> {
    pub attn_qkv_bias: T,
    pub attn_qkv_weight: T,
    pub attn_output_bias: T,
    pub attn_output_weight: T,
    pub attn_norm_bias: T,
    pub attn_norm_weight: T,

    pub ffn_up_bias: T,
    pub ffn_up_weight: T,
    pub ffn_down_bias: T,
    pub ffn_down_weight: T,
    pub ffn_norm_bias: T,
    pub ffn_norm_weight: T,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        let token_embd = &gguf.tensors["token_embd.weight"];
        let position_embd = &gguf.tensors["position_embd.weight"];
        let output_norm_bias = &gguf.tensors["output_norm.bias"];
        let output_norm_weight = &gguf.tensors["output_norm.weight"];
        let output = &gguf.tensors["output.weight"];
        let qkv0 = &gguf.tensors["blk.0.attn_qkv.weight"];
        #[rustfmt::skip]
        let meta = Gpt2Meta {
            dt_embd:  token_embd.ty,
            dt_token_embd:  token_embd.ty,
            dt_postion_embd: position_embd.ty,
            dt_norm: output_norm_weight.ty,
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
                attn_qkv_bias:      gguf.tensors[&*format!("blk.{i}.attn_qkv.bias"     )].data,
                attn_qkv_weight:    gguf.tensors[&*format!("blk.{i}.attn_qkv.weight"   )].data,
                attn_output_bias:   gguf.tensors[&*format!("blk.{i}.attn_output.bias"  )].data,
                attn_output_weight: gguf.tensors[&*format!("blk.{i}.attn_output.weight")].data,
                attn_norm_bias:     gguf.tensors[&*format!("blk.{i}.attn_norm.bias"    )].data,
                attn_norm_weight:   gguf.tensors[&*format!("blk.{i}.attn_norm.weight"  )].data,

                ffn_up_bias:        gguf.tensors[&*format!("blk.{i}.ffn_up.bias"       )].data,
                ffn_up_weight:      gguf.tensors[&*format!("blk.{i}.ffn_up.weight"     )].data,
                ffn_down_bias:      gguf.tensors[&*format!("blk.{i}.ffn_down.bias"     )].data,
                ffn_down_weight:    gguf.tensors[&*format!("blk.{i}.ffn_down.weight"   )].data,
                ffn_norm_bias:      gguf.tensors[&*format!("blk.{i}.ffn_norm.bias"     )].data,
                ffn_norm_weight:    gguf.tensors[&*format!("blk.{i}.ffn_norm.weight"   )].data,
            })
            .collect();

        Self {
            meta,
            token_embd: token_embd.data,
            pos_embd: position_embd.data,
            blocks,
            output_norm_bias: output_norm_bias.data,
            output_norm_weight: output_norm_weight.data,
            output: output.data,
        }
    }
}

impl<T> BlkStorage<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> BlkStorage<U> {
        BlkStorage {
            attn_norm_bias: f(self.attn_norm_bias),
            attn_norm_weight: f(self.attn_norm_weight),
            attn_qkv_bias: f(self.attn_qkv_bias),
            attn_qkv_weight: f(self.attn_qkv_weight),
            attn_output_bias: f(self.attn_output_bias),
            attn_output_weight: f(self.attn_output_weight),

            ffn_up_bias: f(self.ffn_up_bias),
            ffn_up_weight: f(self.ffn_up_weight),
            ffn_down_bias: f(self.ffn_down_bias),
            ffn_down_weight: f(self.ffn_down_weight),
            ffn_norm_bias: f(self.ffn_norm_bias),
            ffn_norm_weight: f(self.ffn_norm_weight),
        }
    }

    pub fn as_ref(&self) -> BlkStorage<&T> {
        BlkStorage {
            attn_norm_bias: &self.attn_norm_bias,
            attn_norm_weight: &self.attn_norm_weight,
            attn_qkv_bias: &self.attn_qkv_bias,
            attn_qkv_weight: &self.attn_qkv_weight,
            attn_output_bias: &self.attn_output_bias,
            attn_output_weight: &self.attn_output_weight,

            ffn_up_bias: &self.ffn_up_bias,
            ffn_up_weight: &self.ffn_up_weight,
            ffn_down_bias: &self.ffn_down_bias,
            ffn_down_weight: &self.ffn_down_weight,
            ffn_norm_bias: &self.ffn_norm_bias,
            ffn_norm_weight: &self.ffn_norm_weight,
        }
    }
}

impl<'w> BlkStorage<&'w [u8]> {
    pub fn distribute<U>(
        &self,
        meta: &Gpt2Meta,
        range: impl RangeBounds<usize>,
        count: usize,
        mut f: impl FnMut(usize) -> U,
    ) -> BlkStorage<Contiguous<'w, U>>
    where
        U: DerefMut<Target = [u8]>,
    {
        let range = normalize(range, count);
        let start = range.start;
        let len = range.len();
        assert!(0 < len && range.end <= count);

        fn tensor<'t>(dt: DigitLayout, shape: &[usize], data: &'t [u8]) -> Tensor<&'t [u8]> {
            Tensor::new(dt, shape).map(|size| {
                debug_assert_eq!(size, data.len());
                data
            })
        }

        BlkStorage {
            attn_qkv_bias: borrow(&self.attn_qkv_bias),
            attn_qkv_weight: borrow(&self.attn_qkv_weight),
            attn_output_bias: borrow(&self.attn_output_bias),
            attn_output_weight: borrow(&self.attn_output_weight),
            attn_norm_bias: borrow(&self.attn_norm_bias),
            attn_norm_weight: borrow(&self.attn_norm_weight),

            ffn_up_bias: borrow(&self.ffn_up_bias),
            ffn_up_weight: borrow(&self.ffn_up_weight),
            ffn_down_bias: borrow(&self.ffn_down_bias),
            ffn_down_weight: borrow(&self.ffn_down_weight),
            ffn_norm_bias: borrow(&self.ffn_norm_bias),
            ffn_norm_weight: borrow(&self.ffn_norm_weight),
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
