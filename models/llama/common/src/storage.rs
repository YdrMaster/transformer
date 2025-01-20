use crate::{LlamaBlkWeight, LlamaMeta};
use common::{borrow, own, Contiguous, Distribution};
use gguf::{GGufMetaMapExt, GGufModel};
use std::ops::DerefMut;
use tensor::{rearrange, split, Tensor};

#[derive(Clone)]
pub struct Storage<T> {
    pub meta: LlamaMeta,
    pub token_embd: T,
    pub output_norm: T,
    pub output: T,
    pub blocks: Box<[BlkStorage<T>]>,
}

#[derive(Clone)]
pub struct BlkStorage<T> {
    pub attn_norm: T,
    pub attn_qkv: T,
    pub attn_qkv_bias: T,
    pub attn_o: T,
    pub ffn_norm: T,
    pub ffn_gate_inp: T,
    pub ffn_gate_up: T,
    pub ffn_down: T,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        use gguf::{meta, tensor};
        let arch = meta!(gguf => general_architecture);
        let token_embd = tensor![gguf => "token_embd.weight"];
        let output_norm = tensor![gguf => "output_norm.weight"];
        let qkv0 = tensor![gguf => "blk.0.attn_qkv.weight"];

        let d = meta![gguf => llm_embedding_length];
        let nh = meta![gguf => llm_attention_head_count];

        #[rustfmt::skip]
        let meta = LlamaMeta {
            dt_embd   :  token_embd.ty,
            dt_norm   : output_norm.ty,
            dt_linear :        qkv0.ty,

            attn_qkv_bias: arch == "qwen2",

            nctx    : meta![gguf => llm_context_length         ],
            nvoc    : meta![gguf => tokenizer_ggml_tokens].len(),
            nexp    : meta![gguf => llm_expert_count        ; 0],
            nexp_use: meta![gguf => llm_expert_used_count   ; 0],

            d, nh,
            nblk: meta![gguf => llm_block_count                 ],
            nkvh: meta![gguf => llm_attention_head_count_kv;  nh],
            dh  : meta![gguf => llm_rope_dimension_count; d / nh],
            di  : meta![gguf => llm_feed_forward_length         ],

            epsilon: meta!(gguf => llm_attention_layer_norm_rms_epsilon; 1e-5),
            theta  : meta!(gguf => llm_rope_freq_base                  ; 1e4 ),
        };

        #[rustfmt::skip]
        let blocks = (0..meta.nblk)
            .map(|i| BlkStorage {
                attn_norm    : tensor![gguf => format!("blk.{i}.attn_norm.weight"  )].data,
                attn_qkv     : tensor![gguf => format!("blk.{i}.attn_qkv.weight"   )].data,
                attn_qkv_bias: if !meta.attn_qkv_bias { &[] }
                               else { tensor![gguf => format!("blk.{i}.attn_qkv.bias")].data },
                attn_o       : tensor![gguf => format!("blk.{i}.attn_output.weight")].data,
                ffn_norm     : tensor![gguf => format!("blk.{i}.ffn_norm.weight"   )].data,
                ffn_gate_inp : if !meta.is_moe() { &[] }
                               else              { tensor![gguf => format!("blk.{i}.ffn_gate_inp.weight"    )].data },
                ffn_gate_up  : if !meta.is_moe() { tensor![gguf => format!("blk.{i}.ffn_gate_up.weight"     )].data }
                               else              { tensor![gguf => format!("blk.{i}.ffn_gate_up_exps.weight")].data },
                ffn_down     : if !meta.is_moe() { tensor![gguf => format!("blk.{i}.ffn_down.weight"        )].data }
                               else              { tensor![gguf => format!("blk.{i}.ffn_down_exps.weight"   )].data },
            })
            .collect();

        Self {
            meta,
            token_embd: token_embd.data,
            output_norm: output_norm.data,
            output: gguf.tensors.get("output.weight").unwrap_or(token_embd).data,
            blocks,
        }
    }
}

impl<T> BlkStorage<T> {
    #[rustfmt::skip]
    pub fn into_vec(self) -> Vec<(LlamaBlkWeight, T)> {
        use LlamaBlkWeight as W;
        vec![
            (W::AttnNorm   , self.attn_norm    ),
            (W::AttnQKV    , self.attn_qkv     ),
            (W::AttnQKVBias, self.attn_qkv_bias),
            (W::AttnO      , self.attn_o       ),
            (W::FfnNorm    , self.ffn_norm     ),
            (W::FfnGateInp , self.ffn_gate_inp ),
            (W::FfnGateUp  , self.ffn_gate_up  ),
            (W::FfnDown    , self.ffn_down     ),
        ]
    }
}

impl<T> FromIterator<(LlamaBlkWeight, T)> for BlkStorage<T> {
    #[rustfmt::skip]
    fn from_iter<U>(iter: U) -> Self
    where
        U: IntoIterator<Item = (LlamaBlkWeight, T)>,
    {
        let mut collector: BlkStorage<Option<T>> = BlkStorage {
            attn_norm    : None,
            attn_qkv     : None,
            attn_qkv_bias: None,
            attn_o       : None,
            ffn_norm     : None,
            ffn_gate_inp : None,
            ffn_gate_up  : None,
            ffn_down     : None,
        };
        for (which, data) in iter {
            use LlamaBlkWeight as W;
            match which {
                W::AttnNorm    => collector.attn_norm     = Some(data),
                W::AttnQKV     => collector.attn_qkv      = Some(data),
                W::AttnQKVBias => collector.attn_qkv_bias = Some(data),
                W::AttnO       => collector.attn_o        = Some(data),
                W::FfnNorm     => collector.ffn_norm      = Some(data),
                W::FfnGateInp  => collector.ffn_gate_inp  = Some(data),
                W::FfnGateUp   => collector.ffn_gate_up   = Some(data),
                W::FfnDown     => collector.ffn_down      = Some(data),
            };
        }
        BlkStorage {
            attn_norm    : collector.attn_norm    .unwrap(),
            attn_qkv     : collector.attn_qkv     .unwrap(),
            attn_qkv_bias: collector.attn_qkv_bias.unwrap(),
            attn_o       : collector.attn_o       .unwrap(),
            ffn_norm     : collector.ffn_norm     .unwrap(),
            ffn_gate_inp : collector.ffn_gate_inp .unwrap(),
            ffn_gate_up  : collector.ffn_gate_up  .unwrap(),
            ffn_down     : collector.ffn_down     .unwrap(),
        }
    }
}

impl LlamaMeta {
    pub fn distribute_data<'w, U>(
        &self,
        which: LlamaBlkWeight,
        data: &'w [u8],
        dist: Distribution,
        mut f: impl FnMut(usize) -> U,
    ) -> Contiguous<'w, U>
    where
        U: DerefMut<Target = [u8]>,
    {
        use crate::TensorUsage::Storage as TensorMem;
        use LlamaBlkWeight as W;
        match which {
            W::AttnNorm | W::FfnNorm | W::FfnGateInp => borrow(data),
            _ if dist.is_mono() || data.is_empty() => borrow(data),
            W::AttnQKV => {
                let meta = self.distribute(dist);
                self.distribute_qkv(
                    dist,
                    meta.attn_qkv(TensorMem).map(&mut f),
                    self.attn_qkv(TensorMem).map(|_| data),
                )
            }
            W::AttnQKVBias => {
                let meta = self.distribute(dist);
                self.distribute_qkv(
                    dist,
                    meta.attn_qkv_bias(TensorMem).map(&mut f),
                    self.attn_qkv_bias(TensorMem).map(|_| data),
                )
            }
            W::AttnO => {
                let [start, len, total] = dist.info();
                let o = self.attn_o(TensorMem).map(|_| data);

                let d = o.shape()[1] / total;
                let o = o.slice(1, d * start, 1, d * len);

                let mut o_ = Tensor::new(o.dt(), o.shape()).map(&mut f);
                rearrange(&mut o_, &o);
                own(o_.take())
            }
            W::FfnGateUp => {
                let &LlamaMeta { di, .. } = self;
                let [start, len, total] = dist.info();
                let dist = self.distribute(dist);

                let gu = self.ffn_gate_up(TensorMem).map(|_| data);
                split!(gu => g, u; [di, di] @ 1);

                let di = di / total;

                let g = g.slice(1, di * start, 1, di * len);
                let u = u.slice(1, di * start, 1, di * len);

                let mut ans = dist.ffn_gate_up(TensorMem).map(&mut f);
                {
                    let ans = ans.map_slice_mut();
                    split!(ans => g_, u_; [di * len , di * len] @ 1);
                    let mut g_ = g_;
                    let mut u_ = u_;
                    rearrange(&mut g_, &g);
                    rearrange(&mut u_, &u);
                }
                own(ans.take())
            }
            W::FfnDown => {
                let [start, len, total] = dist.info();
                let down = self.ffn_down(TensorMem).map(|_| data);

                let d = down.shape()[2] / total;
                let down = down.slice(2, d * start, 1, d * len);

                let mut down_ = Tensor::new(down.dt(), down.shape()).map(&mut f);
                rearrange(&mut down_, &down);
                own(down_.take())
            }
        }
    }

    pub fn distribute_qkv<'w, U>(
        &self,
        dist: Distribution,
        dst: Tensor<U>,
        src: Tensor<&'w [u8]>,
    ) -> Contiguous<'w, U>
    where
        U: DerefMut<Target = [u8]>,
    {
        let &LlamaMeta { nh, nkvh, dh, .. } = self;
        let [start, len, total] = dist.info();

        let dq = nh * dh;
        let dkv = nkvh * dh;

        let qkv = src;
        split!(qkv => q, k, v; [dq, dkv, dkv] @ 0);

        let dq = dq / total;
        let dkv = dkv / total;

        let q = q.slice(0, dq * start, 1, dq * len);
        let k = k.slice(0, dkv * start, 1, dkv * len);
        let v = v.slice(0, dkv * start, 1, dkv * len);
        debug_assert!(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());

        let mut ans = dst;
        {
            let ans = ans.map_slice_mut();
            split!(ans => q_, k_, v_; [dq * len , dkv * len, dkv * len] @ 0);
            let mut q_ = q_;
            let mut k_ = k_;
            let mut v_ = v_;
            rearrange(&mut q_, &q);
            rearrange(&mut k_, &k);
            rearrange(&mut v_, &v);
        }
        own(ans.take())
    }
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
