use crate::{Operators, Weights};
use clip::{ClipArgs, ClipMeta, ClipStorage, ClipWorker, Image, Tensor};
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
        dt_embd,

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
        .normalize(dt_embd, image_mean, image_std);
    println!("slice image {:?}", time.elapsed());

    let weights = Weights::new(&storage);
    let mut worker = Worker::new(&Cpu, meta.clone(), weights);

    let whole = slices.whole();
    worker
        .launch(
            ClipArgs {
                raw: whole.to_nchw(),
                pos: pos70(whole.shape(), d_patch).map_slice(),
            },
            &mut [],
            &ThisThread,
        )
        .unwrap();

    if let Some(patches) = slices.patches_nchw() {
        let &[_, 3, h, w] = patches.shape() else {
            unreachable!()
        };
        worker
            .launch(
                ClipArgs {
                    raw: patches.map_slice(),
                    pos: pos70([w, h], d_patch).map_slice(),
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();
    }
}

fn pos70([w, h]: [usize; 2], d_patch: usize) -> Tensor<Blob> {
    let pos_w = w / d_patch;
    let pos_h = h / d_patch;
    let mut bucket_corrds_h = [0; 70];
    let mut bucket_corrds_w = [0; 70];
    for i in 0..pos_w {
        bucket_corrds_w[i] = ((70 * i) as f64 / pos_w as f64) as _;
    }
    for i in 0..pos_h {
        bucket_corrds_h[i] = ((70 * i) as f64 / pos_h as f64) as _;
    }

    let mut ans = Tensor::new(ty::U32, &[pos_w * pos_h]).map(Blob::new);
    let (&mut [], data, &mut []) = (unsafe { ans.get_mut().align_to_mut::<u32>() }) else {
        panic!()
    };

    let f = |i, d| ((70 * i) as f64 / d as f64) as u32;
    for i in 0..pos_h * pos_w {
        data[i] = f(i / pos_w, pos_h) * 70 + f(i % pos_w, pos_w);
    }

    ans
}
