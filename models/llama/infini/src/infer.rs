use crate::{Operators, RandomSample, Weights};
use gguf::GGufModel;
use llama::{ext::ggml_quants::f16, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use operators::{
    infini_rt::{self, Device, DeviceType::DEVICE_CPU},
    random_sample::{KVPair, SampleArgs},
    TopoNode,
};
use regex::Regex;
use std::{
    iter::zip,
    slice::{from_raw_parts, from_raw_parts_mut},
    thread,
};
use test_utils::{test_infer_paralle, Inference, Task, TokenizerAndPrompt, WorkerSeed};

type Worker<'w> = LlamaWorker<Operators, Weights>;

#[test]
fn test_infer() {
    let Some(Inference {
        model,
        devices,
        prompt,
        as_user,
        temperature,
        top_p,
        top_k,
        max_steps,
    }) = Inference::load()
    else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));

    let TokenizerAndPrompt {
        eos,
        tokenizer,
        prompt,
    } = TokenizerAndPrompt::new(&gguf, prompt, as_user);

    let model = LlamaStorage::from_gguf(&gguf);
    println!("{:?}", model.meta);

    let sample_args = SampleArgs::new(temperature, top_p, top_k).expect("invalid sample args");
    println!("{sample_args:?}");

    let devices = devices
        .map(|devices| {
            Regex::new(r"\d+")
                .unwrap()
                .find_iter(&devices)
                .map(|c| c.as_str().parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![0]);
    let lens = vec![1; devices.len()];
    let count = devices.len();
    println!("distribution: {devices:?}");

    infini_rt::init(DEVICE_CPU);
    let (seeds, senders) = WorkerSeed::new(
        devices
            .into_iter()
            .map(|id| Device { ty: DEVICE_CPU, id })
            .collect(),
    );
    thread::scope(|s| {
        let _workers = zip(lens, seeds)
            .enumerate()
            .scan(0, |start, (i, (len, seed))| {
                let range = *start..*start + len;
                *start = range.end;

                let mut meta = model.meta.clone();
                meta.distribute(range.clone(), count);

                let model = &model;
                Some(s.spawn(move || {
                    let WorkerSeed { node, tasks } = seed;
                    let device = node.processor();
                    let stream = device.stream();
                    let weights = Weights::new(model, range, count, &stream);
                    let mut worker = Worker::new(&node, meta.clone(), weights, i == 0);
                    let mut cache = meta
                        .kv_cache(meta.nctx)
                        .map(|size| stream.malloc::<u8>(size));
                    let sin_cos = <Operators as llama::Operators>::build_sin_cos(
                        meta.dt_embd,
                        meta.nctx,
                        meta.dh,
                        &stream,
                    );

                    let sample = RandomSample::new(&node);
                    let indices = RandomSample::build_indices(model.meta.nvoc, &stream);
                    let mut pair = KVPair::new(0, f16::ZERO);
                    let mut pairs = Tensor::kv_pair_vec(1, |size| stream.malloc::<u8>(size));

                    for task in tasks {
                        let Task {
                            nt,
                            pos,
                            embd,
                            next,
                        } = task;
                        let mut embd = meta
                            .embd(nt)
                            .map(|size| stream.from_host(unsafe { from_raw_parts(embd, size) }));
                        let mut logits = meta
                            .logits(if i == 0 { 1 } else { 0 })
                            .map(|size| stream.malloc::<u8>(size));
                        worker
                            .launch(
                                llama::LlamaArgs {
                                    embd: embd.map_slice_mut(),
                                    logits: logits.map_slice_mut(),
                                    sin_cos: sin_cos.map_slice(),
                                    requests: vec![LlamaRequest {
                                        cache: cache.map_slice_mut(),
                                        seq_len: nt,
                                        out_len: if i == 0 { 1 } else { 0 },
                                        pos,
                                    }],
                                    num_tokens: nt,
                                    max_seq_len: nt,
                                    max_att_len: nt + pos,
                                },
                                &mut [],
                                &stream,
                            )
                            .unwrap();
                        if i == 0 {
                            sample
                                .launch(
                                    &mut pairs,
                                    &logits,
                                    &indices,
                                    sample_args,
                                    &mut [],
                                    &stream,
                                )
                                .unwrap();

                            stream.synchronize();
                            device.memcpy_d2h(
                                unsafe {
                                    from_raw_parts_mut(
                                        &mut pair as *mut _ as *mut u8,
                                        pairs.get().len(),
                                    )
                                },
                                pairs.get(),
                            );

                            next.send(pair.idx() as _).unwrap()
                        }
                    }
                }))
            })
            .collect::<Vec<_>>();

        let senders = senders.into_boxed_slice();
        test_infer_paralle(&model, senders, eos, tokenizer, &prompt, max_steps)
    })
}
