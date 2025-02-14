use crate::{MiniCPM3BlkWeight, MiniCPM3Meta};
use common::{borrow, own, Contiguous, Distribution};
use gguf::{GGufMetaMapExt, GGufModel};
use std::ops::DerefMut;
use tensor::{rearrange, split, Tensor};

#[derive(Clone)]
pub struct Storage<T> {
    pub meta: MiniCPM3Meta,
    pub token_embd: T,
    pub output_norm: T,
    pub output: T,
    pub rope_long: T,
    pub rope_short: T,
    pub blocks: Box<[BlkStorage<T>]>,
}

#[derive(Clone)]
pub struct BlkStorage<T> {
    pub attn_norm: T,
    pub attn_qb: T,
    pub attn_qa: T,
    pub attn_kvb: T,
    pub attn_kva: T,
    pub attn_qa_norm: T,
    pub attn_kva_norm: T,
    pub attn_o: T,
    pub ffn_norm: T,
    pub ffn_gate_up: T,
    pub ffn_gate: T,
    pub ffn_up: T,
    pub ffn_down: T,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        use gguf::{meta, tensor};
        assert_eq!(meta!(gguf => general_architecture), "minicpm3");
        let token_embd = tensor![gguf => "token_embd.weight"];
        let output_norm = tensor![gguf => "output_norm.weight"];
        let rope_long = tensor![gguf => "rope_factors_long.weight"];
        let rope_short = tensor![gguf => "rope_factors_short.weight"];
        let attn_o0 = tensor![gguf => "blk.0.attn_output.weight"];
        let dv = tensor![gguf => format!("blk.0.attn_kv_b.weight"      )].shape[1];
        let dk = meta![gguf => (usize) "minicpm3.attention.key_length"  ];
        let d = meta![gguf => llm_embedding_length];
        let nh = meta![gguf => llm_attention_head_count];

        let dh = meta![gguf => llm_rope_dimension_count; d / nh];
        #[rustfmt::skip]
        let meta = MiniCPM3Meta {
            dt_embd   :  token_embd.ty,
            dt_norm   : output_norm.ty,
            dt_linear :     attn_o0.ty,

            nctx    : meta![gguf => llm_context_length         ],
            nvoc    : meta![gguf => tokenizer_ggml_tokens].len(),
            d, nh,dh,
            nblk: meta![gguf => llm_block_count                 ],
            nkvh: meta![gguf => llm_attention_head_count_kv;  nh],
            di  : meta![gguf => llm_feed_forward_length         ],

            dq_lora: meta![gguf => (usize) "minicpm3.attention.q_lora_rank" ],
            dkv_lora:meta![gguf => (usize) "minicpm3.attention.kv_lora_rank"],
            dk,
            dv:(dv/nh)-dk+dh,
            dnope: dk-dh,
            epsilon: meta!(gguf => llm_attention_layer_norm_rms_epsilon; 1e-5),
            theta  : meta!(gguf => llm_rope_freq_base                  ; 1e4 ),
        };
        #[rustfmt::skip]
        let blocks = (0..meta.nblk)
            .map(|i| BlkStorage {
                attn_norm     : tensor![gguf => format!("blk.{i}.attn_norm.weight"     )].data,
                attn_qb      : tensor![gguf => format!("blk.{i}.attn_q_b.weight"       )].data,
                attn_qa      : tensor![gguf => format!("blk.{i}.attn_q_a.weight"       )].data,
                attn_kvb     : tensor![gguf => format!("blk.{i}.attn_kv_b.weight"      )].data,
                attn_kva : tensor![gguf => format!("blk.{i}.attn_kv_a_mqa.weight"  )].data,
                attn_qa_norm : tensor![gguf => format!("blk.{i}.attn_q_a_norm.weight"  )].data,
                attn_kva_norm: tensor![gguf => format!("blk.{i}.attn_kv_a_norm.weight" )].data,
                attn_o        : tensor![gguf => format!("blk.{i}.attn_output.weight"   )].data,
                ffn_norm      : tensor![gguf => format!("blk.{i}.ffn_norm.weight"      )].data,
                // TODO 待修改 gguf 字段名称应该为 ffn_gate_up
                ffn_gate_up   : tensor![gguf => format!("blk.{i}.attn_output.weight"   )].data,
                ffn_gate:    tensor![gguf => format!("blk.{i}.ffn_gate.weight"       )].data,
                ffn_up:  tensor![gguf => format!("blk.{i}.ffn_up.weight"   )].data,
                ffn_down      : tensor![gguf => format!("blk.{i}.ffn_down.weight"      )].data,
            })
            .collect();

        Self {
            meta,
            token_embd: token_embd.data,
            output_norm: output_norm.data,
            output: gguf.tensors.get("output.weight").unwrap_or(token_embd).data,
            blocks,
            rope_long: rope_long.data,
            rope_short: rope_short.data,
        }
    }
}

impl<T> BlkStorage<T> {
    #[rustfmt::skip]
    pub fn into_vec(self) -> Vec<(MiniCPM3BlkWeight, T)> {
        use MiniCPM3BlkWeight as W;
        vec![
            (W::AttnNorm    , self.attn_norm    ),
            (W::AttnQB      , self.attn_qb      ),
            (W::AttnQA      , self.attn_qa      ),
            (W::AttnKvB     , self.attn_kvb     ),
            (W::AttnKvA      , self.attn_kva     ),
            (W::AttnQANorm  , self.attn_qa_norm ),
            (W::AttnKvANorm , self.attn_kva_norm),
            (W::AttnO      , self.attn_o       ),
            (W::FfnNorm    , self.ffn_norm     ),
            (W::FfnGateUp  , self.ffn_gate_up  ),
            (W::FfnDown    , self.ffn_down     ),
            (W::FfnGate    , self.ffn_gate     ),
            (W::FfnUp    , self.ffn_up     ),
        ]
    }
}

impl<T> FromIterator<(MiniCPM3BlkWeight, T)> for BlkStorage<T> {
    #[rustfmt::skip]
    fn from_iter<U>(iter: U) -> Self
    where
        U: IntoIterator<Item = (MiniCPM3BlkWeight, T)>,
    {
        let mut collector: BlkStorage<Option<T>> = BlkStorage {
            attn_norm     : None,
            attn_o        : None,
            ffn_norm      : None,
            ffn_gate_up   : None,
            ffn_gate: None,
            ffn_up: None,
            ffn_down      : None,
            attn_qb      : None,
            attn_qa      : None,
            attn_kvb     : None,
            attn_kva : None,
            attn_qa_norm : None,
            attn_kva_norm: None,
        };
        for (which, data) in iter {
            use MiniCPM3BlkWeight as W;
            match which {
                W::AttnNorm    => collector.attn_norm     = Some(data),
                W::AttnQB      => collector.attn_qb       = Some(data),
                W::AttnQA      => collector.attn_qa       = Some(data),
                W::AttnKvB     => collector.attn_kvb      = Some(data),
                W::AttnKvA  => collector.attn_kva      = Some(data),
                W::AttnQANorm  => collector.attn_qa_norm  = Some(data),
                W::AttnKvANorm => collector.attn_kva_norm = Some(data),
                W::AttnO       => collector.attn_o        = Some(data),
                W::FfnNorm     => collector.ffn_norm      = Some(data),
                W::FfnGateUp   => collector.ffn_gate_up   = Some(data),
                W::FfnDown     => collector.ffn_down      = Some(data),
                W::FfnGate => collector.ffn_gate   = Some(data),
                W::FfnUp => collector.ffn_up   = Some(data),
            };
        }
        BlkStorage {
            attn_norm    : collector.attn_norm    .unwrap(),
            attn_qb      : collector.attn_qb      .unwrap(),
            attn_qa      : collector.attn_qa      .unwrap(),
            attn_kvb     : collector.attn_kvb     .unwrap(),
            attn_kva     : collector.attn_kva     .unwrap(),
            attn_qa_norm : collector.attn_qa_norm .unwrap(),
            attn_kva_norm: collector.attn_kva_norm.unwrap(),
            attn_o       : collector.attn_o       .unwrap(),
            ffn_norm     : collector.ffn_norm     .unwrap(),
            ffn_gate_up  : collector.ffn_gate_up  .unwrap(),
            ffn_down     : collector.ffn_down     .unwrap(),
            ffn_gate: collector.ffn_gate  .unwrap(),
            ffn_up: collector.ffn_up  .unwrap(),
        }
    }
}

impl MiniCPM3Meta {
    pub fn distribute_data<'w, U>(
        &self,
        which: MiniCPM3BlkWeight,
        data: &'w [u8],
        dist: Distribution,
        mut f: impl FnMut(usize) -> U,
    ) -> Contiguous<'w, U>
    where
        U: DerefMut<Target = [u8]>,
    {
        use crate::TensorUsage::Storage as TensorMem;
        use MiniCPM3BlkWeight as W;
        match which {
            W::AttnQB
            | W::AttnQA
            | W::AttnKvB
            | W::AttnKvA
            | W::AttnQANorm
            | W::AttnKvANorm
            | W::FfnGate
            | W::FfnUp => borrow(data),
            W::AttnNorm | W::FfnNorm => borrow(data),
            _ if dist.is_mono() || data.is_empty() => borrow(data),
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
                let &MiniCPM3Meta { di, .. } = self;
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
        let &MiniCPM3Meta { nh, nkvh, dh, .. } = self;
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
    std::env::set_var(
        "TEST_MODEL",
        "/home/ztf/cpm/Origin-MiniCPM3-4B-v0.0-F16.gguf",
    );
    let Some(Inference { model, .. }) = Inference::load() else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));
    let storage = Storage::from_gguf(&gguf);
    println!("{:#?}", storage.meta);
}
