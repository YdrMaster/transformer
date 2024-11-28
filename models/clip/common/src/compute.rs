use super::{args::Args, ClipMeta};
use operators::{
    conv::{self, Conv},
    ByteOf, Hardware, LaunchError, Operator, QueueAlloc, QueueOf, TopoNode,
};
use std::{
    ops::{Deref, DerefMut},
    time::Instant,
};
use tensor::Tensor;

pub trait Operators {
    type Hardware: Hardware;
    type TopoNode: TopoNode<Self::Hardware>;
    type Conv: Conv<Self::Hardware>;
}

pub trait WeightLoader {
    type Hardware: Hardware;
    type Weight<'s>: Deref<Target = [ByteOf<Self::Hardware>]> + 's
    where
        Self: 's;

    fn patch_embd<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> [Self::Weight<'a>; 2];
}

pub struct ClipWorker<Ops: Operators, W> {
    meta: ClipMeta,
    weights: WeightDecorator<W>,
    conv: Ops::Conv,
    pub debug: bool,
}

impl<Ops: Operators, W> ClipWorker<Ops, W> {
    pub fn new(node: &Ops::TopoNode, meta: ClipMeta, weights: W) -> Self {
        let processor = node.processor();
        Self {
            weights: meta.decorator(weights),
            meta,
            conv: Ops::Conv::new(processor),
            debug: true,
        }
    }

    #[inline]
    pub const fn meta(&self) -> &ClipMeta {
        &self.meta
    }
}

impl<Ops, W> ClipWorker<Ops, W>
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
        let time = Instant::now();
        let Args { raw, .. } = args;
        let queue = queue_alloc.queue();

        let ClipMeta { dt_embd, .. } = self.meta;

        let [k, b] = self.weights.patch_embd(queue);
        let &[n, _, h, w] = raw.shape() else {
            unreachable!()
        };
        let &[m, _, hk, wk] = k.shape() else {
            unreachable!()
        };

        let mut embd = Tensor::new(dt_embd, &[n, m, h / hk, w / wk]).map(|s| queue_alloc.alloc(s));
        self.conv(&mut embd, &raw, &k, &b, workspace, queue_alloc)?;

        let _embd = embd.merge(2..4).unwrap().transpose(&[2, 1]);

        if self.debug {
            println!("encode {n} x {h} x {w} image in {:?}", time.elapsed());
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
impl<Ops, W> ClipWorker<Ops, W>
where
    Ops: Operators,
    W: WeightLoader<Hardware = Ops::Hardware>,
{
    fn conv<Y, X, W_, B, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        w: &Tensor<W_>,
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
        self.conv.launch(
            &conv::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                w_layout: w.layout(),
                w_base: w.base(),
                b_layout: b.layout(),
                b_base: b.base(),
                strides: [self.meta.d_patch; 2],
                dilations: [1; 2],
                pads: [0; 4],
            },
            workspace,
            queue_alloc,
        )
    }
}

struct WeightDecorator<W> {
    weights: W,
    patch_embd_w: Tensor<usize>,
    patch_embd_b: Tensor<usize>,
}

impl ClipMeta {
    fn decorator<W>(&self, weights: W) -> WeightDecorator<W> {
        WeightDecorator {
            patch_embd_w: self.patch_embd_w(),
            patch_embd_b: self.patch_embd_b(),
            weights,
        }
    }
}

impl<W: WeightLoader> WeightDecorator<W> {
    #[inline]
    pub fn patch_embd<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> [Tensor<W::Weight<'a>>; 2] {
        let [w, b] = self.weights.patch_embd(queue);
        [
            self.patch_embd_w.clone().map(|_| w),
            self.patch_embd_b.clone().map(|_| b),
        ]
    }
}
