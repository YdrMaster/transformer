use super::{args::Args, LlamaMeta};
use gguf::ggml_quants::digit_layout::{types as ty, DigitLayout};
use itertools::izip;
use operators::{
    all_reduce::{self, AllReduce, ReduceOp},
    attention_kv_cached::{self, AttnKVCached},
    mat_mul::{self, MatMul},
    mlp::{self, Mlp},
    rearrange::{self, Rearrange},
    rms_norm::{self, RmsNorm},
    rope::{self, Rope, Seq, SinCosTable},
    ByteOf, Hardware, LaunchError, Operator, QueueAlloc, QueueOf, TopoNode, Workspace,
};
use std::ops::{Deref, DerefMut};
use tensor::{split, Tensor};

pub trait Operators {
    type Hardware: Hardware;
    type TopoNode: TopoNode<Self::Hardware>;
    type RmsNorm: RmsNorm<Self::Hardware>;
    type MatMul: MatMul<Self::Hardware>;
    type Rope: Rope<Self::Hardware>;
    type AttnKVCached: AttnKVCached<Self::Hardware>;
    type Mlp: Mlp<Self::Hardware>;
    type Rearrange: Rearrange<Self::Hardware>;
    type AllReduce: AllReduce<Self::Hardware, Self::TopoNode>;

    fn debug<T>(tensor: &Tensor<T>, queue: &QueueOf<Self::Hardware>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>;

    fn build_sin_cos<QA>(
        dt: DigitLayout,
        nctx: usize,
        dh: usize,
        queue_alloc: &QA,
    ) -> Tensor<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let SinCosTable { nctx, mem } =
            <Self::Rope as Rope<Self::Hardware>>::build_sincos(dt, nctx, dh, queue_alloc);
        Tensor::new(dt, &[2, nctx, dh]).map(|_| mem)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlkWeight {
    AttnNorm,
    AttnQKV,
    AttnO,
    FfnNorm,
    FfnGateUp,
    FfnDown,
}

pub trait WeightLoader {
    type Hardware: Hardware;
    type Weight<'s>: Deref<Target = [ByteOf<Self::Hardware>]> + 's
    where
        Self: 's;

    fn load_blk<'a>(
        &'a self,
        which: BlkWeight,
        iblk: usize,
        queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a>;

    fn output_norm<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a>;
    fn output<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a>;
}

pub struct LlamaWorker<Ops: Operators, W> {
    #[allow(dead_code)]
    id: usize,
    meta: LlamaMeta,
    weights: WeightDecorator<W>,
    rms_norm: Ops::RmsNorm,
    mat_mul: Ops::MatMul,
    rope: Ops::Rope,
    attn_kv_cached: Ops::AttnKVCached,
    mlp: Ops::Mlp,
    rearrange: Ops::Rearrange,
    all_reduce: Ops::AllReduce,
    residual: bool,
}

impl<Ops: Operators, W> LlamaWorker<Ops, W> {
    pub fn new(
        id: usize,
        node: &Ops::TopoNode,
        meta: LlamaMeta,
        weights: W,
        residual: bool,
    ) -> Self {
        let processor = node.processor();
        Self {
            id,
            weights: meta.decorator(weights),
            meta,
            rms_norm: Ops::RmsNorm::new(processor),
            mat_mul: Ops::MatMul::new(processor),
            rope: Ops::Rope::new(processor),
            attn_kv_cached: Ops::AttnKVCached::new(processor),
            mlp: Ops::Mlp::new(processor),
            rearrange: Ops::Rearrange::new(processor),
            all_reduce: Ops::AllReduce::new(node),
            residual,
        }
    }

    #[inline]
    pub const fn meta(&self) -> &LlamaMeta {
        &self.meta
    }

    pub fn workspace_size(&self, nt: usize, max_seq_len: usize, max_att_len: usize) -> usize {
        let LlamaMeta {
            dt_mat,
            nh,
            nkvh,
            d,
            dh,
            di,
            ..
        } = self.meta;

        let ele = dt_mat.nbytes();
        let embd = nt * d * ele;
        let qkv = nt * (nh + nkvh + nkvh) * dh * ele;
        let gate_up = nt * di * 2 * ele;
        let q = max_seq_len * nh * dh * ele;
        let att = nkvh * max_seq_len * max_att_len * ele;

        embd + qkv.max(gate_up) + q + att
    }
}

impl<Ops, W> LlamaWorker<Ops, W>
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
            embd,
            sin_cos,
            mut logits,
            mut requests,
            num_tokens: nt,
            max_seq_len,
            max_att_len,
        } = args;
        let LlamaMeta {
            dt_embd,
            nblk,
            nh,
            nkvh,
            dh,
            ..
        } = self.meta;
        let beta = if self.residual { 1. } else { 0. };

        let workspace_size = self.workspace_size(nt, max_seq_len, max_att_len);
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size);

        let mut x = embd;
        let x1 = Tensor::new(dt_embd, x.shape());
        let (buf, workspace) = workspace.split_at_mut(*x1.get());
        let mut x1 = x1.map(|_| buf);

        let qkv = Tensor::new(dt_embd, &[nt, (nh + nkvh + nkvh) * dh]);

        let sin = sin_cos.clone().index(0, 0);
        let cos = sin_cos.index(0, 1);

        let pos = Tensor::new(ty::U64, &[nt]).map(|_| {
            Ops::Rope::build_pos(
                ty::U64,
                nt,
                requests.iter().map(|req| Seq {
                    pos: req.pos,
                    len: req.seq_len,
                }),
                queue_alloc,
            )
        });

        let req_split = requests.iter().map(|req| req.seq_len).collect::<Vec<_>>();

        let queue = queue_alloc.queue();
        for iblk in 0..nblk {
            {
                let w = self.weights.attn_norm(iblk, queue);
                self.rms_norm(&mut x1, &x, &w, workspace, queue_alloc)?;
                drop(w);

                let (buf, workspace) = workspace.split_at_mut(*qkv.get());
                let mut qkv = qkv.clone().map(|_| buf);

                let w = self.weights.attn_qkv(iblk, queue);
                self.mat_mul(&mut qkv, 0., &x1, &w, 1., workspace, queue_alloc)?;
                drop(w);

                let qkv = qkv.tile(1, &[nh + nkvh + nkvh, dh]);

                split!(qkv => q, k, v; [nh, nkvh, nkvh] @ 1);
                let mut q = q;
                let mut k = k;
                let v = v;

                self.rope(&mut q, &pos, &sin, &cos, workspace, queue_alloc)?;
                self.rope(&mut k, &pos, &sin, &cos, workspace, queue_alloc)?;

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

                let o = q.merge(1..3).unwrap();
                let w = self.weights.attn_o(iblk, queue);
                self.mat_mul(&mut x, beta, &o, &w, 1., workspace, queue_alloc)?;
                drop(w);

                self.all_reduce(&mut x, workspace, queue_alloc)?;
            }
            {
                let w = self.weights.ffn_norm(iblk, queue);
                self.rms_norm(&mut x1, &x, &w, workspace, queue_alloc)?;
                drop(w);

                self.mlp(&mut x, &x1, iblk, self.residual, workspace, queue_alloc)?;

                self.all_reduce(&mut x, workspace, queue_alloc)?;
            }
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
        let x_ = unsafe { x.map_slice_static() };
        let w = self.weights.output_norm(queue);
        self.rms_norm(&mut x, &x_, &w, workspace, queue_alloc)?;
        drop(w);

        let w = self.weights.output(queue);
        self.mat_mul(&mut logits, 0., &x, &w, 1., workspace, queue_alloc)
    }
}

#[allow(clippy::too_many_arguments)]
impl<Ops, W> LlamaWorker<Ops, W>
where
    Ops: Operators,
    W: WeightLoader<Hardware = Ops::Hardware>,
{
    fn rms_norm<Y, X, W_, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        w: &Tensor<W_>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        W_: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.rms_norm.launch(
            &rms_norm::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                w_layout: w.layout(),
                w_base: w.base(),
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

    fn rope<T, P, Sin, Cos, QA>(
        &self,
        t: &mut Tensor<T>,
        p: &Tensor<P>,
        sin: &Tensor<Sin>,
        cos: &Tensor<Cos>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        T: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        P: Deref<Target = [ByteOf<Ops::Hardware>]>,
        Sin: Deref<Target = [ByteOf<Ops::Hardware>]>,
        Cos: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.rope.launch(
            &rope::Args {
                t_layout: t.layout(),
                t_base: t.base_mut(),
                p_layout: p.layout(),
                p_base: p.base(),
                sin_layout: sin.layout(),
                sin_base: sin.base(),
                cos_layout: cos.layout(),
                cos_base: cos.base(),
                theta: self.meta.theta,
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

    fn mlp<Y, X, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        iblk: usize,
        residual: bool,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        let queue = queue_alloc.queue();
        let w_gate_up = self.weights.ffn_gate_up(iblk, queue);
        let w_down = self.weights.ffn_down(iblk, queue);

        self.mlp.launch(
            &mlp::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                w_gate_up_layout: w_gate_up.layout(),
                w_gate_up_base: w_gate_up.base(),
                w_down_layout: w_down.layout(),
                w_down_base: w_down.base(),
                down_alpha: 1.,
                residual,
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
}

struct WeightDecorator<W> {
    attn_norm: Tensor<usize>,
    attn_qkv: Tensor<usize>,
    attn_o: Tensor<usize>,
    ffn_norm: Tensor<usize>,
    ffn_gate_up: Tensor<usize>,
    ffn_down: Tensor<usize>,
    output_norm: Tensor<usize>,
    output: Tensor<usize>,
    weights: W,
}

impl LlamaMeta {
    fn decorator<W>(&self, weights: W) -> WeightDecorator<W> {
        use crate::TensorUsage::Computation;
        WeightDecorator {
            attn_norm: self.attn_norm(),
            attn_qkv: self.attn_qkv(Computation),
            attn_o: self.attn_o(Computation),
            ffn_norm: self.ffn_norm(),
            ffn_gate_up: self.ffn_gate_up(Computation),
            ffn_down: self.ffn_down(Computation),
            output_norm: self.output_norm(),
            output: self.output(),
            weights,
        }
    }
}

impl<W: WeightLoader> WeightDecorator<W> {
    #[inline]
    pub fn attn_norm<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        self.attn_norm
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::AttnNorm, iblk, queue))
    }

    #[inline]
    pub fn attn_qkv<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        self.attn_qkv
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::AttnQKV, iblk, queue))
    }

    #[inline]
    pub fn attn_o<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        self.attn_o
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::AttnO, iblk, queue))
    }

    #[inline]
    pub fn ffn_norm<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        self.ffn_norm
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::FfnNorm, iblk, queue))
    }

    #[inline]
    pub fn ffn_gate_up<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        self.ffn_gate_up
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::FfnGateUp, iblk, queue))
    }

    #[inline]
    pub fn ffn_down<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        self.ffn_down
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::FfnDown, iblk, queue))
    }

    #[inline]
    pub fn output_norm<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Weight<'a>> {
        self.output_norm
            .clone()
            .map(|_| self.weights.output_norm(queue))
    }

    #[inline]
    pub fn output<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Weight<'a>> {
        self.output.clone().map(|_| self.weights.output(queue))
    }
}
