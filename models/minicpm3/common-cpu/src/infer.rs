use crate::{Operators, RandomSample, Weights};
use common::Distribution;
use gguf::GGufModel;
use minicpm3::{ext::ggml_quants::f16, MiniCPM3Request, MiniCPM3Storage, Minicpm3Worker, Tensor};
use operators::{
    all_reduce::common_cpu::Operator as AllReduce,
    common_cpu::{InprocNode, ThisThread},
    random_sample::{KVPair, SampleArgs},
    Blob,
};
use regex::Regex;
use std::{
    iter::zip,
    ptr::copy_nonoverlapping,
    slice::from_raw_parts_mut,
    sync::{Arc, Barrier},
    thread,
};
use test_utils::{test_infer_paralle, Inference, Task, TokenizerAndPrompt, WorkerSeed};

type Worker<'w> = Minicpm3Worker<Operators<InprocNode<usize>, AllReduce>, Weights<'w>>;

#[test]
fn test_infer() {
    std::env::set_var(
        "TEST_MODEL",
        "/home/ztf/cpm/Origin-MiniCPM3-4B-v0.0-F16.gguf",
    );
    let Some(Inference {
        model,
        devices,
        mut prompt,
        as_user,
        temperature,
        top_p,
        top_k,
        max_steps,
    }) = Inference::load()
    else {
        return;
    };
    prompt = "我".to_owned();
    let gguf = GGufModel::read(model.iter().map(|s| &**s));

    let TokenizerAndPrompt {
        eos,
        tokenizer,
        prompt,
    } = TokenizerAndPrompt::new(&gguf, prompt, as_user);

    let model = MiniCPM3Storage::from_gguf(&gguf);
    println!("{:?}", model.meta);

    let sample_args = SampleArgs::new(temperature, top_p, top_k).expect("invalid sample args");
    println!("{sample_args:?}");

    let lens = devices
        .map(|devices| {
            Regex::new(r"\d+")
                .unwrap()
                .find_iter(&devices)
                .map(|c| c.as_str().parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![1]);
    let dist = lens.iter().sum();
    println!("distribution: {lens:?}");

    let (seeds, senders) = WorkerSeed::new(InprocNode::new(lens.len()));
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
                    let weights = Weights::new(model, dist);
                    let mut worker = Worker::new(id, &node, meta.clone(), weights);
                    let mut cache = meta.kv_cache(meta.nctx).map(Blob::new);
                    let sin_cos = <Operators as minicpm3::Operators>::build_sin_cos(
                        meta.dt_embd,
                        meta.nctx,
                        meta.dh,
                        &ThisThread,
                    );

                    let sample = RandomSample::new(&node);
                    let indices = RandomSample::build_indices(model.meta.nvoc, &ThisThread);
                    let mut pair = KVPair::new(0, f16::ZERO);
                    let mut pairs = Tensor::kv_pair_vec(1, |_| unsafe {
                        from_raw_parts_mut(&mut pair as *mut _ as *mut u8, size_of_val(&pair))
                    });

                    barrier.wait();
                    for task in tasks {
                        let Task {
                            nt,
                            pos,
                            embd,
                            next,
                        } = task;
                        let mut embd = meta.embd(nt).map(|size| {
                            let mut blob = Blob::new(size);
                            unsafe { copy_nonoverlapping(embd, blob.as_mut_ptr(), size) };
                            blob
                        });
                        let mut logits = meta.logits(if id == 0 { 1 } else { 0 }).map(Blob::new);
                        worker
                            .launch(
                                minicpm3::MiniCPM3Args {
                                    embd: embd.map_slice_mut(),
                                    logits: logits.map_slice_mut(),
                                    sin_cos: sin_cos.map_slice(),
                                    requests: vec![MiniCPM3Request {
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
                                &ThisThread,
                            )
                            .unwrap();
                        if id == 0 {
                            sample
                                .launch(
                                    &mut pairs,
                                    &logits,
                                    &indices,
                                    sample_args,
                                    &mut [],
                                    &ThisThread,
                                )
                                .unwrap();
                            next.send(pair.idx() as _).unwrap()
                        }
                    }
                }))
            })
            .collect::<Vec<_>>();

        let senders = senders.into_boxed_slice();
        barrier.wait();
        test_infer_paralle(
            senders,
            test_utils::AboutToken {
                tokenizer,
                token_embd: model.token_embd,
                nvoc: model.meta.nvoc,
                eos,
            },
            &prompt,
            max_steps,
        )
    })
}
