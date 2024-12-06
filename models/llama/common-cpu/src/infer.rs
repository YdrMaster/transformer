use crate::{Operators, RandomSample, Weights};
use gguf::GGufModel;
use llama::{ext::ggml_quants::f16, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use operators::{
    all_reduce::common_cpu::Operator as AllReduce,
    common_cpu::{InprocNode, ThisThread},
    random_sample::{KVPair, SampleArgs},
    Blob,
};
use regex::Regex;
use std::{iter::zip, ptr::copy_nonoverlapping, slice::from_raw_parts_mut, thread};
use test_utils::{test_infer_paralle, Inference, Task, TokenizerAndPrompt, WorkerSeed};

type Worker<'w> = LlamaWorker<Operators<InprocNode<usize>, AllReduce>, Weights<'w>>;

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

    let lens = devices
        .map(|devices| {
            Regex::new(r"\d+")
                .unwrap()
                .find_iter(&devices)
                .map(|c| c.as_str().parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![1]);
    let count = lens.iter().sum();
    println!("distribution: {lens:?}");

    let (seeds, senders) = WorkerSeed::new(InprocNode::new(lens.len()));
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
                    let weights = Weights::new(model, range, count);
                    let mut worker = Worker::new(&node, meta.clone(), weights, i == 0);
                    let mut cache = meta.kv_cache(meta.nctx).map(Blob::new);
                    let sin_cos = <Operators as llama::Operators>::build_sin_cos(
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
                        let mut logits = meta.logits(if i == 0 { 1 } else { 0 }).map(Blob::new);
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
                                &ThisThread,
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
        test_infer_paralle(&model, senders, eos, tokenizer, &prompt, max_steps)
    })
}
