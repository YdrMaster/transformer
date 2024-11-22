use crate::{normalize, LlamaMeta};
use common::{borrow, own, Contiguous};
use gguf::{GGufMetaMapExt, GGufModel};
use std::ops::{DerefMut, RangeBounds};
use tensor::{rearrange, split, Tensor};

#[derive(Clone)]
pub struct Storage<T> {
    pub meta: LlamaMeta,
    pub token_embd: T,
    pub output_norm: T,
    pub output: T,
    pub blocks: Box<[BlkStorage<T>]>,
}

#[derive(Clone, Copy)]
pub struct BlkStorage<T> {
    pub attn_norm: T,
    pub attn_qkv: T,
    pub attn_o: T,
    pub ffn_norm: T,
    pub ffn_gate_up: T,
    pub ffn_down: T,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        use gguf::GGufMetaError::NotExist;

        let token_embd = &gguf.tensors["token_embd.weight"];
        let output_norm = &gguf.tensors["output_norm.weight"];
        let output = gguf.tensors.get("output.weight");
        let qkv0 = &gguf.tensors["blk.0.attn_qkv.weight"];
        #[rustfmt::skip]
        let meta = LlamaMeta {
            dt_embd:  token_embd.ty,
            dt_norm: output_norm.ty,
            dt_mat :        qkv0.ty,

            nblk: gguf.llm_block_count            ().unwrap(),
            nctx: gguf.llm_context_length         ().unwrap(),
            nvoc: gguf.tokenizer_ggml_tokens      ().unwrap().len(),
            nh  : gguf.llm_attention_head_count   ().unwrap(),
            nkvh: gguf.llm_attention_head_count_kv().unwrap(),
            d   : gguf.llm_embedding_length       ().unwrap(),
            dh  : gguf.llm_rope_dimension_count   ().unwrap(),
            di  : gguf.llm_feed_forward_length    ().unwrap(),

            epsilon: match gguf.llm_attention_layer_norm_rms_epsilon() {
                Ok(val) => val,
                Err(NotExist) => 1e-5,
                Err(e) => panic!("failed to read meta: {e:?}"),
            },
            theta  : match gguf.llm_rope_freq_base() {
                Ok(val) => val,
                Err(NotExist) => 1e4,
                Err(e) => panic!("failed to read meta: {e:?}"),
            },
        };

        #[rustfmt::skip]
        let blocks = (0..meta.nblk)
            .map(|i| BlkStorage {
                attn_norm:   gguf.tensors[&*format!("blk.{i}.attn_norm.weight"  )].data,
                attn_qkv:    gguf.tensors[&*format!("blk.{i}.attn_qkv.weight"   )].data,
                attn_o:      gguf.tensors[&*format!("blk.{i}.attn_output.weight")].data,
                ffn_norm:    gguf.tensors[&*format!("blk.{i}.ffn_norm.weight"   )].data,
                ffn_gate_up: gguf.tensors[&*format!("blk.{i}.ffn_gate_up.weight")].data,
                ffn_down:    gguf.tensors[&*format!("blk.{i}.ffn_down.weight"   )].data,
            })
            .collect();

        Self {
            meta,
            token_embd: token_embd.data,
            output_norm: output_norm.data,
            output: output.unwrap_or(token_embd).data,
            blocks,
        }
    }
}

impl<T> BlkStorage<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> BlkStorage<U> {
        BlkStorage {
            attn_norm: f(self.attn_norm),
            attn_qkv: f(self.attn_qkv),
            attn_o: f(self.attn_o),
            ffn_norm: f(self.ffn_norm),
            ffn_gate_up: f(self.ffn_gate_up),
            ffn_down: f(self.ffn_down),
        }
    }

    pub fn as_ref(&self) -> BlkStorage<&T> {
        BlkStorage {
            attn_norm: &self.attn_norm,
            attn_qkv: &self.attn_qkv,
            attn_o: &self.attn_o,
            ffn_norm: &self.ffn_norm,
            ffn_gate_up: &self.ffn_gate_up,
            ffn_down: &self.ffn_down,
        }
    }
}

impl<'w> BlkStorage<&'w [u8]> {
    pub fn distribute<U>(
        &self,
        meta: &LlamaMeta,
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

        let &LlamaMeta {
            nh, nkvh, dh, di, ..
        } = meta;
        assert_eq!(nkvh % count, 0);
        assert_eq!(di % count, 0);

        let mut dis = meta.clone();
        dis.distribute(range.clone(), count);

        use crate::TensorUsage::Storage as TensorMem;
        BlkStorage {
            attn_norm: borrow(self.attn_norm),
            attn_qkv: if len == count {
                borrow(self.attn_qkv)
            } else {
                let dq = nh * dh;
                let dkv = nkvh * dh;

                let qkv = meta.attn_qkv(TensorMem).map(|_| self.attn_qkv);
                split!(qkv => q, k, v; [dq, dkv, dkv] @ 0);

                let dq = dq / count;
                let dkv = dkv / count;

                let q = q.slice(0, dq * start, 1, dq * len);
                let k = k.slice(0, dkv * start, 1, dkv * len);
                let v = v.slice(0, dkv * start, 1, dkv * len);
                debug_assert!(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());

                let mut ans = dis.attn_qkv(TensorMem).map(&mut f);
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
            },
            attn_o: if len == count {
                borrow(self.attn_o)
            } else {
                let o = meta.attn_o(TensorMem).map(|_| self.attn_o);

                let d = o.shape()[1] / count;
                let o = o.slice(1, d * start, 1, d * len);

                let mut o_ = Tensor::new(o.dt(), o.shape()).map(&mut f);
                rearrange(&mut o_, &o);
                own(o_.take())
            },
            ffn_norm: borrow(self.ffn_norm),
            ffn_gate_up: if len == count {
                borrow(self.ffn_gate_up)
            } else {
                let gu = meta.ffn_gate_up(TensorMem).map(|_| self.ffn_gate_up);
                split!(gu => g, u; [di, di] @ 0);

                let di = di / count;

                let g = g.slice(0, di * start, 1, di * len);
                let u = u.slice(0, di * start, 1, di * len);
                debug_assert!(g.is_contiguous() && u.is_contiguous());

                let mut ans = dis.ffn_gate_up(TensorMem).map(&mut f);
                {
                    let ans = ans.map_slice_mut();
                    split!(ans => g_, u_; [di * len , di * len] @ 0);
                    let mut g_ = g_;
                    let mut u_ = u_;
                    rearrange(&mut g_, &g);
                    rearrange(&mut u_, &u);
                }
                own(ans.take())
            },
            ffn_down: if len == count {
                borrow(self.ffn_down)
            } else {
                let down = meta.ffn_down(TensorMem).map(|_| self.ffn_down);

                let d = down.shape()[1] / count;
                let down = down.slice(1, d * start, 1, d * len);

                let mut down_ = Tensor::new(down.dt(), down.shape()).map(&mut f);
                rearrange(&mut down_, &down);
                own(down_.take())
            },
        }
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
