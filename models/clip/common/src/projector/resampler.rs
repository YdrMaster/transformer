use gguf::{tensor, GGufMetaMapExt, GGufModel};

#[derive(Clone, Debug)]
pub struct Meta {
    pub d: usize,
    pub dq: usize,
    pub dh: usize,
}

impl Meta {
    pub fn from_gguf(gguf: &GGufModel) -> Self {
        match gguf.get_usize("clip.minicpmv_version").unwrap() {
            2 => Self {
                d: 4096,
                dq: 96,
                dh: 128,
            },
            3 | 4 => Self {
                d: 3584,
                dq: 64,
                dh: 128,
            },
            version => todo!("Unsupported MiniCPM version: {version}"),
        }
    }
}

#[derive(Clone)]
pub struct Storage<T> {
    pub wkv: T,
    pub q: T,
    pub ln_q: [T; 2],
    pub ln_kv: [T; 2],
    pub attn_q: [T; 2],
    pub attn_k: [T; 2],
    pub attn_v: [T; 2],
    pub attn_o: [T; 2],
    pub ln_post: [T; 2],
    pub proj: T,
}

impl<'a> Storage<&'a [u8]> {
    #[rustfmt::skip]
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        Self {
            wkv    :  tensor![gguf => "resampler.kv.weight"      ].data ,
            q      :  tensor![gguf => "resampler.query"          ].data ,
            ln_q   : [tensor![gguf => "resampler.ln_q.weight"    ].data ,
                      tensor![gguf => "resampler.ln_q.bias"      ].data],
            ln_kv  : [tensor![gguf => "resampler.ln_kv.weight"   ].data ,
                      tensor![gguf => "resampler.ln_kv.bias"     ].data],
            attn_q : [tensor![gguf => "resampler.attn.q.weight"  ].data ,
                      tensor![gguf => "resampler.attn.q.bias"    ].data],
            attn_k : [tensor![gguf => "resampler.attn.k.weight"  ].data ,
                      tensor![gguf => "resampler.attn.k.bias"    ].data],
            attn_v : [tensor![gguf => "resampler.attn.v.weight"  ].data ,
                      tensor![gguf => "resampler.attn.v.bias"    ].data],
            attn_o : [tensor![gguf => "resampler.attn.out.weight"].data ,
                      tensor![gguf => "resampler.attn.out.bias"  ].data],
            ln_post: [tensor![gguf => "resampler.ln_post.weight" ].data ,
                      tensor![gguf => "resampler.ln_post.bias"   ].data],
            proj   :  tensor![gguf => "resampler.proj.weight"    ].data ,
        }
    }
}
