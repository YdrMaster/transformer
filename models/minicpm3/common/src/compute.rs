use super::{args::Args, MiniCPM3BlkWeight, MiniCPM3Meta};
use gguf::ggml_quants::digit_layout::types as ty;
use gguf::ggml_quants::digit_layout::DigitLayout;
use half::f16;
use itertools::Itertools;
use operators::scale;
use operators::scale::Scale;
use operators::{
    add::{self, Add},
    all_reduce::{self, AllReduce, ReduceOp},
    attention::{self, Attention},
    attention_kv_cached::{AttnKVCached},
    fuesd_softmax::AttnMask,
    mat_mul::{self, MatMul},
    rearrange::{self, Rearrange},
    rms_norm::{self, RmsNorm},
    rope::{self, Rope, SinCosTable},
    swiglu::{self, Swiglu},
    ByteOf, Hardware, LaunchError, Operator, QueueAlloc, QueueOf, TopoNode, Workspace,
};
use std::ops::{Deref, DerefMut};
use tensor::split_mut;
use tensor::{split, Tensor};

pub trait Operators {
    type Hardware: Hardware;
    type TopoNode: TopoNode<Self::Hardware>;
    type Attention: Attention<Self::Hardware>;
    type AttnKVCached: AttnKVCached<Self::Hardware>;
    type Rope: Rope<Self::Hardware>;
    type RmsNorm: RmsNorm<Self::Hardware>;
    type Add: Add<Self::Hardware>;
    type MatMul: MatMul<Self::Hardware>;
    type Swiglu: Swiglu<Self::Hardware>;
    type Scale: Scale<Self::Hardware>;
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

pub trait WeightLoader {
    type Hardware: Hardware;
    type Weight<'s>: Deref<Target = [ByteOf<Self::Hardware>]> + 's
    where
        Self: 's;

    fn load_blk<'a>(
        &'a self,
        which: MiniCPM3BlkWeight,
        iblk: usize,
        queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a>;

    fn output_norm<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a>;
    fn output<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a>;
    fn long_factor<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a>;
    fn short_factor<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a>;
}

pub struct Minicpm3Worker<Ops: Operators, W> {
    id: usize,
    meta: MiniCPM3Meta,
    weights: WeightDecorator<W>,
    dt_pos: DigitLayout,
    add: Ops::Add,
    attn_kv_cached: Ops::AttnKVCached,
    attention: Ops::Attention,
    rope: Ops::Rope,
    rms_norm: Ops::RmsNorm,
    mat_mul: Ops::MatMul,
    scale: Ops::Scale,
    swiglu: Ops::Swiglu,
    rearrange: Ops::Rearrange,
    all_reduce: Ops::AllReduce,
}

impl<Ops: Operators, W> Minicpm3Worker<Ops, W> {
    pub fn new(id: usize, node: &Ops::TopoNode, meta: MiniCPM3Meta, weights: W) -> Self {
        let processor = node.processor();
        Self {
            id,
            weights: meta.decorator(weights),
            meta,
            attn_kv_cached: Ops::AttnKVCached::new(processor),
            rope: Ops::Rope::new(processor),
            rms_norm: Ops::RmsNorm::new(processor),
            mat_mul: Ops::MatMul::new(processor),
            scale: Ops::Scale::new(processor),
            swiglu: Ops::Swiglu::new(processor),
            rearrange: Ops::Rearrange::new(processor),
            add: Ops::Add::new(processor),
            all_reduce: Ops::AllReduce::new(node),
            dt_pos: ty::U64,
            attention: Ops::Attention::new(processor),
        }
    }

    #[inline]
    pub const fn meta(&self) -> &MiniCPM3Meta {
        &self.meta
    }
}

impl<Ops, W> Minicpm3Worker<Ops, W>
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
            embd: mut x,
            logits,
            requests,
            num_tokens: nt,
            sin_cos,
            ..
        } = args;
        let MiniCPM3Meta {
            nblk,
            di,
            dq_lora,
            nh,
            dk,
            dh,
            dkv_lora,
            dv,
            dt_embd,
            
            ..
        } = self.meta;
        // llama.cpp 定义死
        let scale_emb = 12f32;
        let scale_depth = 1.4f32;
        //  残差连接时权重缩放
        let s = scale_depth / (nblk as f32).sqrt();

        let dnope = dk - dh;
        let tensor = |shape: &[usize]| Tensor::new(dt_embd, shape);
        let x1 = tensor(x.shape());

        let gate_up = tensor(&[nt, di * 2]);
        // 空间 x+x1+q(应该可以删除)+q3+kv_pe+attn
        let workspace_size = *x1.get() * 3 + *gate_up.get();
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size);
        let (buf, workspace) = workspace.split_at_mut(*x1.get());
        let mut x1 = x1.map(|_| buf);

        // 经行 attention
        let attn = tensor(&[nt, nh, dv]);
        let (buf, workspace) = workspace.split_at_mut(*attn.get());
        let mut attn = attn.map(|_| buf);

        let queue = queue_alloc.queue();
        // 缩放
        let inplace = unsafe { x.map_slice_static() };
        self.scale(&mut x, &inplace, scale_emb, workspace, queue_alloc)?;
        for iblk in 0..nblk {
            // norm
            let w = self.weights.attn_norm(iblk, queue);
            self.rms_norm(&mut x1, &x, &w, workspace, queue_alloc)?;
            drop(w);
            let q = tensor(&[nt, dq_lora]);
            let (buf, workspace) = workspace.split_at_mut(*q.get());
            let mut q = q.map(|_| buf);
            let w = self.weights.attn_qa(iblk, queue).transpose(&[1, 0]);
            self.mat_mul(&mut q, 0., &x1, &w, 1., workspace, queue_alloc)?;

            let inplace = unsafe { q.map_slice_static() };
            let w = self.weights.attn_qa_norm(iblk, queue);
            self.rms_norm(&mut q, &inplace, &w, workspace, queue_alloc)?;
            {
                // q [1, 768] q1 [1, 3840]   kv_pe [1,288]  kv  [1, 5120] k  [1, 3840] attn  [1, 2560]
                let q1 = tensor(&[nt, nh * dk]);
                let (buf, workspace) = workspace.split_at_mut(*q1.get());
                let mut q1 = q1.map(|_| buf);
                let w = self.weights.attn_qb(iblk, queue).transpose(&[1, 0]);
                self.mat_mul(&mut q1, 0., &q, &w, 1., workspace, queue_alloc)?;
                drop(q);
                // q3 是计算 attn 需要用到的数据，但是我们仍然需要对 q3 的的部分进行嵌入操作
                let mut q3 = q1.tile(1, &[nh, dk]);
                let q2 = unsafe { q3.map_slice_static_mut() };
                split_mut!(q2=>_q, q_rope;[dnope, dh]@ 2);

                // kv_pe [1,288]
                let kv_pe = tensor(&[nt, dkv_lora + dh]);
                let (buf, workspace) = workspace.split_at_mut(*kv_pe.get());
                let mut kv_pe = kv_pe.map(|_| buf);

                let w = self.weights.attn_kva(iblk, queue).transpose(&[1, 0]);
                self.mat_mul(&mut kv_pe, 0., &x1, &w, 1., workspace, queue_alloc)?;

                split_mut!(kv_pe =>  kv_lora, k_rope; [dkv_lora, dh] @ 1);

                let inplace = unsafe { kv_lora.map_slice_static() };
                let w = self.weights.attn_kva_norm(iblk, queue);
                self.rms_norm(&mut kv_lora, &inplace, &w, workspace, queue_alloc)?;
                //   kv    X[1, 5120]
                let kv = tensor(&[nt, nh * (dnope + dv)]);
                let (buf, workspace) = workspace.split_at_mut(*kv.get());
                let mut kv = kv.map(|_| buf);
                let w = self.weights.attn_kvb(iblk, queue).transpose(&[1, 0]);

                self.mat_mul(&mut kv, 0., &kv_lora, &w, 1., workspace, queue_alloc)?;

                let kv = kv.tile(1, &[nh, dnope + dv]);

                split_mut!(kv =>  k_nope ,v ; [dnope  , dv ] @ 2);

                /// longrope
                pub fn longrope(
                    embd: &mut [f32],
                    pos: f32,
                    theta: f32,
                    long_factor: &[f32],
                    short_factor: &[f32],
                    max_pos: f32,
                    origin_max_pos: f32,
                ) {
                    use std::slice::from_raw_parts_mut;
                    // 计算 scaling_factor
                    let scaling_factor =
                        1.0 + ((max_pos / origin_max_pos).ln() / origin_max_pos.ln()).sqrt();
                    let factor = if pos > origin_max_pos {
                        long_factor
                    } else {
                        short_factor
                    };
                    let dh = embd.len() / 2;
                    let embd =
                        unsafe { from_raw_parts_mut(embd.as_mut_ptr().cast::<[f32; 2]>(), dh) };
                    for (i, pair) in embd.iter_mut().enumerate() {
                        let theta = theta.powf(-(i as f32 / dh as f32));
                        let freq = pos * theta * factor.get(i).unwrap().recip();
                        let (sin, cos) = freq.sin_cos();
                        let (sin, cos) = (sin * scaling_factor, cos * scaling_factor);
                        let [a, b] = *pair;
                        *pair = [a * cos - b * sin, a * sin + b * cos];
                    }
                }
                let cast = |t: *const f32| -> &'static [f32] {
                    unsafe { std::slice::from_raw_parts(t, dh / 2) }
                };
                let [long_factor, short_factor] = self.weights.factor(queue);
                let long_factor = cast(long_factor.base().cast());
                let short_factor = cast(short_factor.base().cast());

                //  k   [1, 3840]
                let k = tensor(&[nt, nh, dk]);
                let (buf, workspace) = workspace.split_at_mut(*k.get());
                let k = k.map(|_| buf);

                split_mut!(k =>  k_nope_r ,k_rope_r ; [dnope, dh] @ 2);

                let pos = requests.last().unwrap().pos as f32;
                let (max_pos, origin_max_pos) = (100f32, 100f32);

                // q 嵌入
                (0..nh).for_each(|i| {
                    let tmp_q = unsafe {
                        std::slice::from_raw_parts_mut(
                            q_rope.base_mut().cast::<f32>().add(i * 32),
                            32,
                        )
                    };
                    longrope(
                        tmp_q,
                        pos,
                        self.meta.theta,
                        long_factor,
                        short_factor,
                        max_pos,
                        origin_max_pos,
                    );
                });
                //  k 嵌入

                let k_rope_1 =
                    unsafe { std::slice::from_raw_parts_mut(k_rope.base_mut().cast::<f32>(), 32) };
                longrope(
                    k_rope_1,
                    pos,
                    self.meta.theta,
                    long_factor,
                    short_factor,
                    max_pos,
                    origin_max_pos,
                );

                // 经行广播和拷贝
                let k_rope = k_rope.tile(1, &[1, dh]).broadcast(1, nh);
                self.rearrange(&mut k_rope_r, &k_rope, workspace, queue_alloc)?;
                self.rearrange(&mut k_nope_r, &k_nope, workspace, queue_alloc)?;

                let mut q = q3.transpose(&[1, 0]);
                let k = k.map_slice().transpose(&[1, 0]);
                let v = v.map_slice_mut().transpose(&[1, 0]);
                let mut attn = unsafe { attn.map_slice_mut().transpose(&[1, 0]) };
                self.attnention(
                    &mut q,
                    &k,
                    &v,
                    &mut attn,
                    pos as usize,
                    workspace,
                    queue_alloc,
                )?;

                let o = attn.transpose(&[1, 0]).merge(1..3).unwrap();
                let w = self.weights.attn_o(iblk, queue);

                self.mat_mul(&mut x1, 0., &o, &w, s, workspace, queue_alloc)?;
                let inplace = unsafe { x.map_slice_static() };
                self.add(&mut x, &inplace, &x1, workspace, queue_alloc)?;
            }
            let w = self.weights.ffn_norm(iblk, queue);
            self.rms_norm(&mut x1, &x, &w, workspace, queue_alloc)?;
            drop(w);

            let (buf, workspace) = workspace.split_at_mut(*gate_up.get());
            let gate_up = gate_up.clone().map(|_| buf);
            split!(gate_up => gate, up; [di, di] @ 1);
            let mut gate = gate;
            let mut up = up;
            let w = self.weights.ffn_gate(iblk, queue);
            self.mat_mul(&mut gate, 0., &x1, &w, 1., workspace, queue_alloc)?;

            let w = self.weights.ffn_up(iblk, queue);
            self.mat_mul(&mut up, 0., &x1, &w, 1., workspace, queue_alloc)?;

            self.swiglu(&mut gate, &up, workspace, queue_alloc)?;

            fn print_first_10_elements(ptr: *const f16) {
                assert!(!ptr.is_null(), "Pointer must not be null");

                unsafe {
                    for i in 0..10 {
                        // 逐个访问并打印前 10 个元素
                        let element = ptr.offset(i as isize).read();
                        println!("Element {}: {:?}", i, element);
                    }
                }
            }

            let w = self.weights.ffn_down(iblk, queue);
            self.mat_mul(&mut x1, 0., &gate, &w, s, workspace, queue_alloc)?;

            let inplace = unsafe { x.map_slice_static() };
            self.add(&mut x, &inplace, &x1, workspace, queue_alloc)?;

            self.all_reduce(&mut x, workspace, queue_alloc)?
        }
        if logits.shape()[0] == 0 {
            return Ok(());
        }
        Ops::debug(&x, queue);
        todo!();
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
                    self.rearrange(&mut dst, &src, workspace, queue_alloc)?
                }
                dst += 1
            }
        }
        assert_eq!(dst, logits.shape()[0]);

        let mut x = x.map_slice_mut().slice(0, 0, 1, dst);
        {
            let inplace = unsafe { x.map_slice_static() };
            let w = self.weights.output_norm(queue);
            self.rms_norm(&mut x, &inplace, &w, workspace, queue_alloc)?
        }
        let w = self.weights.output(queue);
        self.mat_mul(&mut logits, 0., &x, &w, 1., workspace, queue_alloc)
    }
}

#[allow(clippy::too_many_arguments)]
impl<Ops, W> Minicpm3Worker<Ops, W>
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
    fn attnention<Q, K, V, O, QA>(
        &self,
        q: &mut Tensor<Q>,
        k: &Tensor<K>,
        v: &Tensor<V>,
        o: &mut Tensor<O>,
        pos: usize,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Q: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        K: Deref<Target = [ByteOf<Ops::Hardware>]>,
        V: Deref<Target = [ByteOf<Ops::Hardware>]>,
        O: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.attention.launch(
            &attention::Args {
                q_layout: q.layout(),
                q_base: q.base_mut(),
                k_layout: k.layout(),
                k_base: k.base(),
                v_layout: v.layout(),
                v_base: v.base(),
                o_layout: o.layout(),
                o_base: o.base_mut(),
                mask: AttnMask::Causal,
            },
            workspace,
            queue_alloc,
        )
    }

    fn swiglu<Gate, Up, QA>(
        &self,
        gate: &mut Tensor<Gate>,
        up: &Tensor<Up>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Gate: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        Up: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.swiglu.launch(
            &swiglu::Args {
                gate_layout: gate.layout(),
                gate_base: gate.base_mut(),
                up_layout: up.layout(),
                up_base: up.base(),
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

    fn add<C, A, B, QA>(
        &self,
        c: &mut Tensor<C>,
        a: &Tensor<A>,
        b: &Tensor<B>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        C: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        A: Deref<Target = [ByteOf<Ops::Hardware>]>,
        B: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.add.launch(
            &add::Args {
                c_layout: c.layout(),
                c_base: c.base_mut(),
                a_layout: a.layout(),
                a_base: a.base(),
                b_layout: b.layout(),
                b_base: b.base(),
            },
            workspace,
            queue_alloc,
        )
    }
    fn scale<C, A, QA>(
        &self,
        c: &mut Tensor<C>,
        a: &Tensor<A>,
        s: f32,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        C: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        A: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.scale.launch(
            &scale::Args {
                c_layout: c.layout(),
                c_base: c.base_mut(),
                a_layout: a.layout(),
                a_base: a.base(),
                s,
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
    norm: Tensor<usize>,
    attn_qb: Tensor<usize>,
    attn_qa: Tensor<usize>,
    attn_kvb: Tensor<usize>,
    attn_kva_mqa: Tensor<usize>,
    attn_qa_norm: Tensor<usize>,
    attn_kva_norm: Tensor<usize>,
    attn_o: Tensor<usize>,
    ffn_gate_up: Tensor<usize>,
    ffn_down: Tensor<usize>,
    factor: [Tensor<usize>; 2],
    ffn_gate: Tensor<usize>,
    ffn_up: Tensor<usize>,
    output: Tensor<usize>,
    weights: W,
}

impl MiniCPM3Meta {
    fn decorator<W>(&self, weights: W) -> WeightDecorator<W> {
        use crate::TensorUsage::Computation;
        WeightDecorator {
            norm: self.norm(),
            attn_qa: self.attn_qa(Computation),
            attn_qb: self.attn_qb(Computation),
            attn_kvb: self.attn_kvb(Computation),
            attn_kva_mqa: self.attn_kva(Computation),
            attn_qa_norm: self.attn_qa_norm(),
            attn_kva_norm: self.attn_kva_norm(),
            attn_o: self.attn_o(Computation),
            ffn_gate_up: self.ffn_gate_up(Computation),
            ffn_down: self.ffn_down(Computation),
            ffn_gate: self.ffn(Computation),
            ffn_up: self.ffn(Computation),
            factor: self.factor(),
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
        let w = self
            .weights
            .load_blk(MiniCPM3BlkWeight::AttnNorm, iblk, queue);
        self.norm.clone().map(|_| w)
    }
    #[inline]
    pub fn attn_qa<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        let w = self
            .weights
            .load_blk(MiniCPM3BlkWeight::AttnQA, iblk, queue);
        self.attn_qa.clone().map(|_| w)
    }
    #[inline]
    pub fn attn_qb<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        let w = self
            .weights
            .load_blk(MiniCPM3BlkWeight::AttnQB, iblk, queue);
        self.attn_qb.clone().map(|_| w)
    }

    #[inline]
    pub fn attn_kvb<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        let w = self
            .weights
            .load_blk(MiniCPM3BlkWeight::AttnKvB, iblk, queue);
        self.attn_kvb.clone().map(|_| w)
    }
    pub fn attn_kva<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        let w = self
            .weights
            .load_blk(MiniCPM3BlkWeight::AttnKvA, iblk, queue);
        self.attn_kva_mqa.clone().map(|_| w)
    }
    #[inline]
    pub fn attn_qa_norm<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        let w = self
            .weights
            .load_blk(MiniCPM3BlkWeight::AttnQANorm, iblk, queue);
        self.attn_qa_norm.clone().map(|_| w)
    }
    #[inline]
    pub fn attn_kva_norm<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        let w = self
            .weights
            .load_blk(MiniCPM3BlkWeight::AttnKvANorm, iblk, queue);
        self.attn_kva_norm.clone().map(|_| w)
    }
    #[inline]
    pub fn attn_o<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        let w = self.weights.load_blk(MiniCPM3BlkWeight::AttnO, iblk, queue);
        self.attn_o.clone().map(|_| w)
    }

    #[inline]
    pub fn ffn_norm<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        let w = self
            .weights
            .load_blk(MiniCPM3BlkWeight::FfnNorm, iblk, queue);
        self.norm.clone().map(|_| w)
    }

    #[inline]
    pub fn ffn_gate_up<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        const WHICH: MiniCPM3BlkWeight = MiniCPM3BlkWeight::FfnGateUp;
        let w = self.weights.load_blk(WHICH, iblk, queue);
        self.ffn_gate_up.clone().map(|_| w)
    }
    #[inline]
    pub fn ffn_gate<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        const WHICH: MiniCPM3BlkWeight = MiniCPM3BlkWeight::FfnGate;
        let w = self.weights.load_blk(WHICH, iblk, queue);
        self.ffn_gate.clone().map(|_| w)
    }
    #[inline]
    pub fn ffn_up<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        const WHICH: MiniCPM3BlkWeight = MiniCPM3BlkWeight::FfnUp;
        let w = self.weights.load_blk(WHICH, iblk, queue);
        self.ffn_up.clone().map(|_| w)
    }
    #[inline]
    pub fn ffn_down<'a>(
        &'a self,
        iblk: usize,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Tensor<W::Weight<'a>> {
        const WHICH: MiniCPM3BlkWeight = MiniCPM3BlkWeight::FfnDown;
        let w = self.weights.load_blk(WHICH, iblk, queue);
        self.ffn_down.clone().map(|_| w)
    }

    #[inline]
    pub fn output_norm<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Weight<'a>> {
        self.norm.clone().map(|_| self.weights.output_norm(queue))
    }

    #[inline]
    pub fn output<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Weight<'a>> {
        self.output.clone().map(|_| self.weights.output(queue))
    }
    #[inline]
    fn factor<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> [Tensor<W::Weight<'a>>; 2] {
        [
            self.factor[0]
                .clone()
                .map(|_| self.weights.long_factor(queue)),
            self.factor[1]
                .clone()
                .map(|_| self.weights.short_factor(queue)),
        ]
    }
}
