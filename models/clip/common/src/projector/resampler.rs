use gguf::{tensor, GGufMetaMapExt, GGufModel};

#[derive(Clone, Debug)]
pub struct Meta {
    pub d: usize,
    pub dq: usize,
}

impl Meta {
    pub fn from_gguf(gguf: &GGufModel) -> Self {
        match gguf.get_usize("clip.minicpmv_version").unwrap() {
            2 => Self { d: 4096, dq: 96 },
            3 | 4 => Self { d: 3584, dq: 64 },
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
}

impl<'a> Storage<&'a [u8]> {
    #[rustfmt::skip]
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        Self {
            wkv  :  tensor![gguf => "resampler.kv.weight"   ].data ,
            q    :  tensor![gguf => "resampler.query"       ].data ,
            ln_q : [tensor![gguf => "resampler.ln_q.weight" ].data ,
                    tensor![gguf => "resampler.ln_q.bias"   ].data],
            ln_kv: [tensor![gguf => "resampler.ln_kv.weight"].data ,
                    tensor![gguf => "resampler.ln_kv.bias"  ].data],
        }
    }
}
