use crate::{Operators, Weights};
use clip::{ClipArgs, ClipMeta, ClipStorage, ClipWorker, Image};
use gguf::GGufModel;
use operators::common_cpu::{Cpu, ThisThread};
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
            },
            &mut [],
            &ThisThread,
        )
        .unwrap();

    let [x, y] = slices.grid();
    for i in 0..y {
        for j in 0..x {
            let patch = slices.patch(j, i);
            worker
                .launch(
                    ClipArgs {
                        raw: patch.to_nchw(),
                    },
                    &mut [],
                    &ThisThread,
                )
                .unwrap();
        }
    }
}
