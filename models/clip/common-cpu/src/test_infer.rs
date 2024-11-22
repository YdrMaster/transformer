use clip::{ClipMeta, ClipStorage, Image};
use gguf::GGufModel;
use std::time::Instant;
use test_utils::Inference;

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
    let _slices = image
        .slice_uhd(9, d_image, d_patch)
        .normalize(dt_embd, image_mean, image_std);
    println!("slice image {:?}", time.elapsed());
}
