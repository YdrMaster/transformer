use super::{args::Args, MiniCPM3BlkWeight, MiniCPM3Meta};
use gguf::ggml_quants::digit_layout::DigitLayout;
use operators::{
    all_reduce::{self, AllReduce, ReduceOp},
    mat_mul::{self, MatMul},
    rearrange::{self, Rearrange},
    rms_norm::{self, RmsNorm},
    swiglu::{self, Swiglu},
    ByteOf, Hardware, LaunchError, Operator, QueueAlloc, QueueOf, TopoNode, Workspace,
};
use std::ops::{Deref, DerefMut};
use tensor::{split, Tensor};

pub trait Operators {
    type Hardware: Hardware;
    type TopoNode: TopoNode<Self::Hardware>;
    type RmsNorm: RmsNorm<Self::Hardware>;
    type MatMul: MatMul<Self::Hardware>;
    type Swiglu: Swiglu<Self::Hardware>;
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
        todo!()
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
}

pub struct Minicpm3Worker<Ops: Operators, W> {
    id: usize,
    meta: MiniCPM3Meta,
    weights: WeightDecorator<W>,
    rms_norm: Ops::RmsNorm,
    mat_mul: Ops::MatMul,
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
            rms_norm: Ops::RmsNorm::new(processor),
            mat_mul: Ops::MatMul::new(processor),
            swiglu: Ops::Swiglu::new(processor),
            rearrange: Ops::Rearrange::new(processor),
            all_reduce: Ops::AllReduce::new(node),
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
            mut logits,
            requests,
            num_tokens: nt,
            ..
        } = args;
        let MiniCPM3Meta { nblk, di, .. } = self.meta;

        let tensor = |shape: &[usize]| Tensor::new(x.dt(), shape);
        let x1 = tensor(x.shape());
        let gate_up = tensor(&[nt, di * 2]);

        let workspace_size = *x1.get() + *gate_up.get();

        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size);
        let (buf, workspace) = workspace.split_at_mut(*x1.get());
        let mut x1 = x1.map(|_| buf);

        let queue = queue_alloc.queue();
        for iblk in 0..nblk {
            {
                let w = self.weights.attn_norm(iblk, queue);
                self.rms_norm(&mut x1, &x, &w, workspace, queue_alloc)?;
                drop(w);

                let o = x1.map_slice(); // TODO
                let w = self.weights.attn_o(iblk, queue);
                let residual = if self.id == 0 { 1. } else { 0. };
                self.mat_mul(&mut x, residual, &o, &w, 1., workspace, queue_alloc)?
            }
            self.all_reduce(&mut x, workspace, queue_alloc)?;

            let w = self.weights.ffn_norm(iblk, queue);
            self.rms_norm(&mut x1, &x, &w, workspace, queue_alloc)?;
            drop(w);

            let (buf, workspace) = workspace.split_at_mut(*gate_up.get());
            let mut gate_up = gate_up.clone().map(|_| buf);

            let w = self.weights.ffn_gate_up(iblk, queue);
            self.mat_mul(&mut gate_up, 0., &x1, &w, 1., workspace, queue_alloc)?;
            drop(w);

            split!(gate_up => gate, up; [di, di] @ 1);
            let mut gate = gate;
            self.swiglu(&mut gate, &up, workspace, queue_alloc)?;

            let w = self.weights.ffn_down(iblk, queue);
            let residual = if self.id == 0 { 1. } else { 0. };
            self.mat_mul(&mut x, residual, &gate, &w, 1., workspace, queue_alloc)?;
            self.all_reduce(&mut x, workspace, queue_alloc)?
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
    attn_o: Tensor<usize>,
    ffn_gate_up: Tensor<usize>,
    ffn_down: Tensor<usize>,
    output: Tensor<usize>,
    weights: W,
}

impl MiniCPM3Meta {
    fn decorator<W>(&self, weights: W) -> WeightDecorator<W> {
        use crate::TensorUsage::Computation;
        WeightDecorator {
            norm: self.norm(),
            attn_o: self.attn_o(Computation),
            ffn_gate_up: self.ffn_gate_up(Computation),
            ffn_down: self.ffn_down(Computation),
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
}
