use crate::{Operators, Weights};
use clip::{ClipArgs, ClipMeta, ClipStorage, ClipWorker, Image, Tensor, D_POS_EMBD};
use gguf::{
    ggml_quants::{digit_layout::types as ty, f16},
    GGufModel,
};
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
                pos: pos70(whole.shape(), d_patch).map_slice(),
                pos_resampler: pos_resampler(3584, whole.shape(), d_patch).map_slice(),
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
                    pos_resampler: pos_resampler(3584, [w, h], d_patch).map_slice(),
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();
    }
}

fn pos70([w, h]: [usize; 2], d_patch: usize) -> Tensor<Blob> {
    let w = w / d_patch;
    let h = h / d_patch;

    let mut ans = Tensor::new(ty::U32, &[1, h * w]).map(Blob::new);
    let (&mut [], data, &mut []) = (unsafe { ans.get_mut().align_to_mut::<u32>() }) else {
        panic!()
    };

    for i in 0..h * w {
        let r = i / w;
        let c = i % w;

        let y = r * D_POS_EMBD / h;
        let x = c * D_POS_EMBD / w;
        data[i] = (y * D_POS_EMBD + x) as _;
    }

    ans
}

fn pos_resampler(d: usize, [w, h]: [usize; 2], d_patch: usize) -> Tensor<Blob> {
    let w = w / d_patch;
    let h = h / d_patch;

    let mut ans = Tensor::new(ty::F16, &[1, h * w, d]).map(Blob::new);
    let (&mut [], data, &mut []) = (unsafe { ans.get_mut().align_to_mut::<f16>() }) else {
        panic!()
    };

    assert!(d % 4 == 0);
    let cache = sin_cos_cache(w.max(h), d / 4, 1e4);

    for i in 0..h * w {
        let r = i / w;
        let c = i % w;

        let data = &mut data[i * d..][..d];
        let d = d / 4;
        for i in 0..d {
            let (sin, cos) = cache[c * d + i];
            data[0 * d..][i] = f16::from_f32(sin);
            data[1 * d..][i] = f16::from_f32(cos);
            let (sin, cos) = cache[r * d + i];
            data[2 * d..][i] = f16::from_f32(sin);
            data[3 * d..][i] = f16::from_f32(cos);
        }
    }

    ans
}

fn sin_cos_cache(max_idx: usize, d: usize, theta: f32) -> Vec<(f32, f32)> {
    (0..max_idx * d)
        .map(|i| {
            let a = (i / d) as f32;
            let b = (i % d) as f32;
            (a * theta.powf(-(b / d as f32))).sin_cos()
        })
        .collect()
}
