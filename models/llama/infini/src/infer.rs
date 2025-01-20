use crate::{Operators, RandomSample, Weights};
use common::Distribution;
use gguf::{ggml_quants::digit_layout::types, GGufModel};
use llama::{ext::ggml_quants::f16, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use operators::{
    infini::InfiniNode,
    infini_rt,
    random_sample::{KVPair, SampleArgs},
    TopoNode,
};
use regex::Regex;
use std::{
    iter::zip,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::{Arc, Barrier},
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

    let (ty, indices) = devices
        .map(|devices| {
            let (ty, tail) = devices.split_once(';').unwrap();
            (
                ty.to_ascii_lowercase(),
                Regex::new(r"\d+")
                    .unwrap()
                    .find_iter(tail)
                    .map(|c| c.as_str().parse().unwrap())
                    .collect(),
            )
        })
        .unwrap_or_else(|| ("cpu".into(), vec![0]));
    let lens = vec![1; indices.len()];
    let dist = indices.len();
    println!("{ty}; distribution: {indices:?}");

    let (seeds, senders) = match &*ty {
        "cpu" => {
            infini_rt::init(infini_rt::DEVICE_CPU);
            WorkerSeed::new(InfiniNode::cpu(indices.len()))
        }
        "nv" => {
            infini_rt::init(infini_rt::DEVICE_NVIDIA);
            WorkerSeed::new(InfiniNode::nv_gpu(&indices))
        }
        "cambricon" => {
            infini_rt::init(infini_rt::DEVICE_CAMBRICON);
            WorkerSeed::new(InfiniNode::cambricon_mlu(&indices))
        }
        "ascend" => {
            infini_rt::init(infini_rt::DEVICE_ASCEND);
            WorkerSeed::new(InfiniNode::ascend_npu(&indices))
        }
        _ => todo!(),
    };
    let barrier = Arc::new(Barrier::new(dist + 1));
    thread::scope(|s| {
        let _workers = zip(lens, seeds)
            .enumerate()
            .scan(0, |start, (id, (len, seed))| {
                let dist = Distribution::new(*start, len, dist);
                *start += len;

                let meta = model.meta.distribute(dist);
                let model = &model;
                let barrier = barrier.clone();
                Some(s.spawn(move || {
                    let WorkerSeed { node, tasks } = seed;
                    let device = node.processor();
                    let weights = Weights::new(model, dist, device);
                    let mut worker = Worker::new(id, &node, meta.clone(), weights);

                    let stream = device.stream();
                    let mut cache = meta
                        .kv_cache(meta.nctx)
                        .map(|size| stream.malloc::<u8>(size));
                    let sin_cos = <Operators as llama::Operators>::build_sin_cos(
                        types::F32,
                        meta.nctx,
                        meta.dh,
                        &stream,
                    );

                    let sample = RandomSample::new(&node);
                    let indices = RandomSample::build_indices(model.meta.nvoc, &stream);
                    barrier.wait();
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
                            .logits(if id == 0 { 1 } else { 0 })
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
                                        out_len: if id == 0 { 1 } else { 0 },
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
                        if id == 0 {
                            // NOTICE 目前 random sample 完全是 CPU 上执行的，没必要再拷贝了
                            let mut pair = KVPair::new(0, f16::ZERO);
                            let mut pairs = Tensor::kv_pair_vec(1, |_| unsafe {
                                from_raw_parts_mut(&mut pair as *mut _ as _, size_of_val(&pair))
                            });
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
                            next.send(pair.idx() as _).unwrap()
                        }
                    }
                }))
            })
            .collect::<Vec<_>>();

        let senders = senders.into_boxed_slice();
        barrier.wait();
        test_infer_paralle(&model, senders, eos, tokenizer, &prompt, max_steps)
    })
}
