use super::{args::Args, Gpt2Meta};
use itertools::izip;
use operators::{
    add_rows::{self, AddRows},
    all_reduce::{self, AllReduce, ReduceOp},
    attention_kv_cached::{self, AttnKVCached},
    gpt2_mlp::{self, Gpt2Mlp},
    layer_norm::{self, LayerNorm},
    mat_mul::{self, MatMul},
    rearrange::{self, Rearrange},
    ByteOf, Hardware, LaunchError, Operator, QueueAlloc, QueueOf, TopoNode, Workspace,
};
use std::ops::{Deref, DerefMut};
use tensor::{split, Tensor};

pub trait Operators {
    type Hardware: Hardware;
    type TopoNode: TopoNode<Self::Hardware>;
    type LayerNorm: LayerNorm<Self::Hardware>;
    type MatMul: MatMul<Self::Hardware>;
    type AttnKVCached: AttnKVCached<Self::Hardware>;
    type Rearrange: Rearrange<Self::Hardware>;
    type AllReduce: AllReduce<Self::Hardware, Self::TopoNode>;
    type AddRows: AddRows<Self::Hardware>;
    type Mlp: Gpt2Mlp<Self::Hardware>;
    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlkWeight {
    AttnNorm,
    AttnQKV,
    AttnO,
    FfnNorm,
    FfnUp,
    FfnDown,
}

pub trait WeightLoader {
    type Hardware: Hardware;
    type Memory<'s>: Deref<Target = [ByteOf<Self::Hardware>]> + 's
    where
        Self: 's;

    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        queue: &QueueOf<Self::Hardware>,
    ) -> [Self::Memory<'_>; 2];

    fn output_norm(&self, queue: &QueueOf<Self::Hardware>) -> [Self::Memory<'_>; 2];
    fn output(&self, queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_>;
    fn pos_embd<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a>;
}

pub struct Gpt2Worker<Ops: Operators, W> {
    meta: Gpt2Meta,
    weights: WeightDecorator<W>,
    layer_norm: Ops::LayerNorm,
    mat_mul: Ops::MatMul,
    attn_kv_cached: Ops::AttnKVCached,
    rearrange: Ops::Rearrange,
    all_reduce: Ops::AllReduce,
    add_rows: Ops::AddRows,
    mlp: Ops::Mlp,
}

impl<Ops: Operators, W> Gpt2Worker<Ops, W> {
    pub fn new(node: &Ops::TopoNode, meta: Gpt2Meta, weights: W) -> Self {
        let processor = node.processor();
        Self {
            weights: meta.decorator(weights), // meta.decorator
            meta,
            layer_norm: Ops::LayerNorm::new(processor),
            mat_mul: Ops::MatMul::new(processor),
            attn_kv_cached: Ops::AttnKVCached::new(processor),
            rearrange: Ops::Rearrange::new(processor),
            all_reduce: Ops::AllReduce::new(node),
            add_rows: Ops::AddRows::new(processor),
            mlp: Ops::Mlp::new(processor),
        }
    }

    #[inline]
    pub const fn meta(&self) -> &Gpt2Meta {
        &self.meta
    }

    pub fn workspace_size(&self, nt: usize, max_seq_len: usize, max_att_len: usize) -> usize {
        let Gpt2Meta {
            dt_mat,
            nh,
            nkvh,
            d,
            // dh,
            di,
            ..
        } = self.meta;

        let ele = dt_mat.nbytes();
        let embd = nt * d * ele;
        let qkv = nt * (nh + nkvh + nkvh) * ele;
        let gate_up = nt * di * 2 * ele;
        let q = max_seq_len * nh * ele;
        let att = nkvh * max_seq_len * max_att_len * ele;

        embd + qkv.max(gate_up) + q + att
    }
}

impl<Ops, W> Gpt2Worker<Ops, W>
where
    Ops: Operators,
    W: WeightLoader<Hardware = Ops::Hardware>,
    ByteOf<Ops::Hardware>: 'static,
{
    pub fn launch<QA>(
        &mut self,
        args: Args<Ops::Hardware>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        let Args {
            mut token_embd,
            mut logits,
            mut requests,
            num_tokens: nt,
            max_seq_len,
            max_att_len,
            idx,
            idx_add,
        } = args;

        let Gpt2Meta {
            dt_embd,
            nblk,
            nh,
            nkvh,
            dh,
            ..
        } = self.meta;
        let workspace_size = self.workspace_size(nt, max_seq_len, max_att_len);
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size);
        let queue = queue_alloc.queue();
        let old_token_embd_l = token_embd.layout();
        // wpe+wte
        {
            self.add_rows(
                &mut token_embd,
                &self.weights.pos_embd(queue),
                &idx,
                &mut workspace,
                queue_alloc,
            )?;
            token_embd = token_embd.merge(0..2).unwrap();
        }
        let mut x = token_embd;
        let x1 = Tensor::new(dt_embd, x.shape());
        let (buf, workspace) = workspace.split_at_mut(*x1.get());
        let mut x1 = x1.map(|_| buf);
        let qkv = Tensor::new(dt_embd, &[nt, (nh + nkvh + nkvh) * dh]);

        let req_split = requests.iter().map(|req| req.seq_len).collect::<Vec<_>>();

        for iblk in 0..nblk {
            {
                let [scale, bias] = self.weights.attn_norm(iblk, queue);
                self.layer_norm(&mut x1, &x, &scale, &bias, workspace, queue_alloc)?
            }
            let (buf, workspace) = workspace.split_at_mut(*qkv.get());
            let mut qkv = qkv.clone().map(|_| buf);
            {
                let [scale, bias] = self.weights.attn_qkv(iblk, queue);
                let cols = bias.shape()[0];
                let bias = bias.tile(0, &[1, cols]).broadcast(0, nt);
                self.rearrange(&mut qkv, &bias, workspace, queue_alloc)?;
                self.mat_mul(&mut qkv, 1., &x1, &scale, 1., workspace, queue_alloc)?;
            }
            let qkv = qkv.tile(1, &[nh + nkvh + nkvh, dh]);
            split!(qkv => q, k, v; [nh, nkvh, nkvh] @ 1);
            let mut q = q;
            let k = k;
            let v = v;
            {
                let q = q.map_slice_mut().transpose(&[1, 0]);
                let k = k.map_slice().transpose(&[1, 0]);
                let v = v.map_slice().transpose(&[1, 0]);
                let q = q.split(1, &req_split);
                let k = k.split(1, &req_split);
                let v = v.split(1, &req_split);

                for (mut q, k, v, req) in izip!(q, k, v, &mut requests) {
                    let cache = req
                        .cache
                        .as_mut() // [buf, nblk, 2, nkvh, dh]
                        .index(1, iblk) // [buf, 2, nkvh, dh]
                        .transpose(&[2, 0]) // [nkvh, 2, buf, dh]
                        .map(|t| &mut t[..]);

                    split!(cache => kc, vc; [1, 1] @ 1);
                    let mut o = unsafe { q.map_slice_static_mut() };
                    self.attn_kv_cached(
                        &mut q,
                        &k,
                        &v,
                        &mut o,
                        &mut kc.index(1, 0),
                        &mut vc.index(1, 0),
                        req.pos,
                        workspace,
                        queue_alloc,
                    )?;
                }
            }
            {
                let o = q.map_slice().merge(1..3).unwrap();
                let [scale, bias] = self.weights.attn_o(iblk, queue);
                let cols = bias.shape()[0];
                let bias = bias.tile(0, &[1, cols]).broadcast(0, nt);
                self.rearrange(&mut x1, &bias, workspace, queue_alloc)?;
                self.mat_mul(&mut x1, 1., &o, &scale, 1., workspace, queue_alloc)?;
            }
            self.all_reduce(&mut x1, workspace, queue_alloc)?;

            // 残差连接 wte+wpe 的数据
            self.add_rows.launch(
                &add_rows::Args {
                    dst_layout: old_token_embd_l.clone(),
                    dst_base: x1.base_mut(),
                    src_layout: x.layout(),
                    src_base: x.base(),
                    idx_layout: idx_add.layout(),
                    idx_base: idx_add.map_slice().base(),
                },
                workspace,
                queue_alloc,
            )?;
            {
                let [scale, bias] = self.weights.ffn_norm(iblk, queue);
                self.layer_norm(&mut x, &x1, &scale, &bias, workspace, queue_alloc)?
            }
            self.mlp(&mut x, iblk, workspace, queue_alloc)?;
            // 残差连接 att 之后的数据
            let mut x = x.map_slice_mut().tile(0, &[1, nt]);
            self.add_rows(&mut x, &x1, &idx_add, workspace, queue_alloc)?
        }
        if logits.shape()[0] == 0 {
            return Ok(());
        }

        // 集中要采样的 token
        // NOTICE: 输入之前将请求按 seq len 升序排列可降低移动开销
        let mut dst = 0;
        let mut src = 0;
        for req in &requests {
            src += req.seq_len;
            for src in src - req.out_len..src {
                if src != dst {
                    let src = unsafe { x.map_slice_static() }.index(0, src);
                    let mut dst = x.map_slice_mut().index(0, dst);
                    self.rearrange(&mut dst, &src, workspace, queue_alloc)?;
                }
                dst += 1;
            }
        }
        assert_eq!(dst, logits.shape()[0]);

        let mut x = x.map_slice_mut().slice(0, 0, 1, dst);
        {
            let inplace = unsafe { x.map_slice_static() };
            let [scale, bias] = self.weights.output_norm(queue);
            self.layer_norm(&mut x, &inplace, &scale, &bias, workspace, queue_alloc)?
        }
        let output = self.weights.output_weight(queue).transpose(&[1, 0]);
        self.mat_mul(&mut logits, 0., &x, &output, 1., workspace, queue_alloc)
    }
}

#[allow(clippy::too_many_arguments)]
impl<Ops, W> Gpt2Worker<Ops, W>
where
    Ops: Operators,
    W: WeightLoader<Hardware = Ops::Hardware>,
{
    fn layer_norm<Y, X, W_, B, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        s: &Tensor<W_>,
        b: &Tensor<B>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        W_: Deref<Target = [ByteOf<Ops::Hardware>]>,
        B: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.layer_norm.launch(
            &layer_norm::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                scale_layout: s.layout(),
                scale_base: s.base(),
                bias_layout: b.layout(),
                bias_base: b.base(),
                epsilon: self.meta.epsilon,
            },
            workspace,
            queue_alloc,
        )
    }

    fn mat_mul<C, A, B, QA>(
        &self,
        c: &mut Tensor<C>,
        beta: f32,
        a: &Tensor<A>,
        b: &Tensor<B>,
        alpha: f32,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        C: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        A: Deref<Target = [ByteOf<Ops::Hardware>]>,
        B: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: c.layout(),
                c_base: c.base_mut(),
                beta,
                a_layout: a.layout(),
                a_base: a.base(),
                b_layout: b.layout(),
                b_base: b.base(),
                alpha,
            },
            workspace,
            queue_alloc,
        )
    }

    fn attn_kv_cached<Q, K, V, O, KC, VC, QA>(
        &self,
        q: &mut Tensor<Q>,
        k: &Tensor<K>,
        v: &Tensor<V>,
        o: &mut Tensor<O>,
        kc: &mut Tensor<KC>,
        vc: &mut Tensor<VC>,
        pos: usize,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Q: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        K: Deref<Target = [ByteOf<Ops::Hardware>]>,
        V: Deref<Target = [ByteOf<Ops::Hardware>]>,
        O: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        KC: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        VC: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.attn_kv_cached.launch(
            &attention_kv_cached::Args {
                q_layout: q.layout(),
                q_base: q.base_mut(),
                k_layout: k.layout(),
                k_base: k.base(),
                v_layout: v.layout(),
                v_base: v.base(),
                o_layout: o.layout(),
                o_base: o.base_mut(),
                k_cache_layout: kc.layout(),
                k_cache_base: kc.base_mut(),
                v_cache_layout: vc.layout(),
                v_cache_base: vc.base_mut(),
                pos: pos.into(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn rearrange<Y, X, QA>(
        &self,
        dst: &mut Tensor<Y>,
        src: &Tensor<X>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: dst.layout(),
                dst_base: dst.base_mut(),
                src_layout: src.layout(),
                src_base: src.base(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn all_reduce<X, QA>(
        &self,
        x: &mut Tensor<X>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        X: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.all_reduce.launch(
            &all_reduce::Args {
                pair: rearrange::Args {
                    dst_layout: x.layout(),
                    dst_base: x.base_mut(),
                    src_layout: x.layout(),
                    src_base: x.base(),
                },
                op: ReduceOp::Sum,
            },
            workspace,
            queue_alloc,
        )
    }

    fn add_rows<Dst, Src, Idx, QA>(
        &self,
        dst: &mut Tensor<Dst>,
        src: &Tensor<Src>,
        idx: &Tensor<Idx>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Dst: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        Src: Deref<Target = [ByteOf<Ops::Hardware>]>,
        Idx: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.add_rows.launch(
            &add_rows::Args {
                dst_layout: dst.layout(),
                dst_base: dst.base_mut(),
                src_layout: src.layout(),
                src_base: src.base(),
                idx_layout: idx.layout(),
                idx_base: idx.base(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn mlp<Y, QA>(
        &self,
        y: &mut Tensor<Y>,
        iblk: usize,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        let queue = queue_alloc.queue();
        let [up_weight, up_bias] = self.weights.ffn_up(iblk, queue);
        let [down_weight, down_bias] = self.weights.ffn_down(iblk, queue);
        self.mlp.launch(
            &gpt2_mlp::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                up_weight_layout: up_weight.layout(),
                up_weight_base: up_weight.base(),
                up_bias_layout: up_bias.layout(),
                up_bias_base: up_bias.base(),
                down_weight_layout: down_weight.layout(),
                down_weight_base: down_weight.base(),
                down_bias_layout: down_bias.layout(),
                down_bias_base: down_bias.base(),
            },
            workspace,
            queue_alloc,
        )
    }
}

struct WeightDecorator<W> {
    attn_norm_w: Tensor<usize>,
    attn_norm_b: Tensor<usize>,
    attn_qkv_w: Tensor<usize>,
    attn_qkv_b: Tensor<usize>,
    attn_o_w: Tensor<usize>,
    attn_o_b: Tensor<usize>,

    ffn_norm_w: Tensor<usize>,
    ffn_norm_b: Tensor<usize>,
    ffn_up_w: Tensor<usize>,
    ffn_up_b: Tensor<usize>,
    ffn_down_w: Tensor<usize>,
    ffn_down_b: Tensor<usize>,

    output_norm_w: Tensor<usize>,
    output_norm_b: Tensor<usize>,
    output_weight: Tensor<usize>,
    pos_embd: Tensor<usize>,

    weights: W,
}

impl Gpt2Meta {
    fn decorator<W>(&self, weights: W) -> WeightDecorator<W> {
        use crate::TensorUsage::Computation;
        WeightDecorator {
            attn_norm_w: self.attn_norm_w(),
            attn_norm_b: self.attn_norm_b(),
            attn_qkv_w: self.attn_qkv_w(Computation),
            attn_qkv_b: self.attn_qkv_b(),
            attn_o_w: self.attn_o_w(Computation),
            attn_o_b: self.attn_o_b(),

            ffn_norm_w: self.ffn_norm_w(),
            ffn_norm_b: self.ffn_norm_b(),
            ffn_up_w: self.ffn_up_w(Computation),
            ffn_up_b: self.ffn_up_b(),
            ffn_down_w: self.ffn_down_w(Computation),
            ffn_down_b: self.ffn_down_b(),

            output_norm_w: self.output_norm_w(),
            output_norm_b: self.output_norm_b(),
            output_weight: self.output_weight(),
            pos_embd: self.pos_embd(),

            weights,
        }
    }
}

impl<W: WeightLoader> WeightDecorator<W> {
    pub fn attn_norm(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::AttnNorm, iblk, queue);
        [
            self.attn_norm_w.clone().map(|_| w),
            self.attn_norm_b.clone().map(|_| b),
        ]
    }

    pub fn attn_qkv(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::AttnQKV, iblk, queue);
        [
            self.attn_qkv_w.clone().map(|_| w),
            self.attn_qkv_b.clone().map(|_| b),
        ]
    }

    pub fn attn_o(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::AttnO, iblk, queue);
        [
            self.attn_o_w.clone().map(|_| w),
            self.attn_o_b.clone().map(|_| b),
        ]
    }

    pub fn ffn_norm(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::FfnNorm, iblk, queue);
        [
            self.ffn_norm_w.clone().map(|_| w),
            self.ffn_norm_b.clone().map(|_| b),
        ]
    }

    pub fn ffn_up(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::FfnUp, iblk, queue);
        [
            self.ffn_up_w.clone().map(|_| w),
            self.ffn_up_b.clone().map(|_| b),
        ]
    }

    pub fn ffn_down(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::FfnDown, iblk, queue);
        [
            self.ffn_down_w.clone().map(|_| w),
            self.ffn_down_b.clone().map(|_| b),
        ]
    }

    pub fn output_norm(&self, queue: &QueueOf<W::Hardware>) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.output_norm(queue);
        [
            self.output_norm_w.clone().map(|_| w),
            self.output_norm_b.clone().map(|_| b),
        ]
    }

    pub fn output_weight(&self, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.output_weight
            .clone()
            .map(|_| self.weights.output(queue))
    }

    pub fn pos_embd<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Memory<'a>> {
        let pos_embd = self.weights.pos_embd(queue);
        self.pos_embd.clone().map(|_| pos_embd)
    }
}
