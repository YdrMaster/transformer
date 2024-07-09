use common::{f16, utok, Blob};
use operators::cndrv::{memcpy_d2h, DevByte, Queue};
use sample::SampleArgs;
use tensor::reslice;

pub fn sample_cpu(
    args: impl IntoIterator<Item = (usize, SampleArgs)>,
    logits: &[DevByte],
    voc: usize,
    _queue: &Queue,
) -> Vec<utok> {
    let mut host = Blob::new(logits.len());
    memcpy_d2h(&mut host, logits);

    let logits: &[f16] = reslice(&host);
    args.into_iter()
        .map(|(i, arg)| arg.random(&logits[voc * i..][..voc]))
        .collect()
}
