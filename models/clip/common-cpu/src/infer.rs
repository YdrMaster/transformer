use crate::{Operators, Weights};
use clip::{ClipArgs, ClipMeta, ClipStorage, ClipWorker, Image, Tensor, D_POS_EMBD};
use gguf::{ggml_quants::digit_layout::types as ty, GGufModel};
use operators::{
    common_cpu::{Cpu, ThisThread},
    Blob,
};
use std::time::Instant;
use test_utils::Inference;

type Worker<'w> = ClipWorker<Operators, Weights<'w>>;

#[test]
fn test_infer() {
    let Some(Inference { model, .. }) = Inference::load() else {
        return;
    };
    let Some(picture) = test_utils::image() else {
        return;
    };

    let gguf = GGufModel::read(model.iter().map(|s| &**s));
    let storage = ClipStorage::from_gguf(&gguf);
    let meta = &storage.meta;
    println!("{meta:#?}");

    let &ClipMeta {
        dt,

        d_image,
        d_patch,

        image_mean,
        image_std,
        ..
    } = meta;

    let time = Instant::now();
    let image = Image::load(picture);
    println!("load image {:?}", time.elapsed());

    let time = Instant::now();
    let slices = image
        .slice_uhd(9, d_image, d_patch)
        .normalize(dt, image_mean, image_std);
    println!("slice image {:?}", time.elapsed());

    let weights = Weights::new(&storage);
    let mut worker = Worker::new(&Cpu, meta.clone(), weights);

    let whole = slices.whole();
    worker
        .launch(
            ClipArgs {
                raw: whole.to_nchw(),
                pos: pos70(1, whole.shape(), d_patch).map_slice(),
            },
            &mut [],
            &ThisThread,
        )
        .unwrap();

    if let Some(patches) = slices.patches_nchw() {
        let &[n, 3, h, w] = patches.shape() else {
            unreachable!()
        };
        worker
            .launch(
                ClipArgs {
                    raw: patches.map_slice(),
                    pos: pos70(n, [w, h], d_patch).map_slice(),
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();
    }
}

fn pos70(n: usize, [w, h]: [usize; 2], d_patch: usize) -> Tensor<Blob> {
    let pos_w = w / d_patch;
    let pos_h = h / d_patch;

    let mut ans = Tensor::new(ty::U32, &[1, pos_w * pos_h])
        .broadcast(0, n)
        .map(Blob::new);
    let (&mut [], data, &mut []) = (unsafe { ans.get_mut().align_to_mut::<u32>() }) else {
        panic!()
    };

    for i in 0..pos_h * pos_w {
        let y = (i / pos_w) * D_POS_EMBD / pos_h;
        let x = (i % pos_w) * D_POS_EMBD / pos_w;
        data[i] = (y * D_POS_EMBD + x) as _;
    }

    ans
}
