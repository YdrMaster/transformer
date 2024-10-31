use crate::{Operators, RandomSample, Weights};
use gguf::GGufModel;
use llama::{ext::ggml_quants::f16, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use operators::{
    all_reduce::common_cpu::Operator as AllReduce,
    common_cpu::{Blob, Cpu, InprocNode, ThisThread},
    random_sample::{KVPair, SampleArgs},
};
use std::{
    iter::zip,
    ptr::copy_nonoverlapping,
    slice::from_raw_parts_mut,
    sync::{
        mpsc::{Receiver, Sender},
        Arc, Barrier,
    },
    thread,
};
use test_utils::{Inference, TokenizerAndPrompt};

type Worker<'w> = LlamaWorker<Operators<InprocNode<usize>, AllReduce>, Weights<'w>>;

#[test]
fn test_dist() {
    let Some(Inference {
        model,
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

    let lens = [1; 4];
    let count = lens.iter().sum();
    let (seeds, senders) = WorkerSeed::new(lens.len());
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
                    for task in tasks {
                        let Task {
                            nt,
                            pos,
                            embd,
                            logits,
                            barrier,
                        } = task;
                        let mut embd = meta.embd(nt).map(|size| {
                            let mut blob = Blob::new(size);
                            unsafe { copy_nonoverlapping(embd, blob.as_mut_ptr(), size) };
                            blob
                        });
                        let mut logits = if i == 0 {
                            meta.logits(1)
                                .map(|size| unsafe { from_raw_parts_mut(logits, size) })
                        } else {
                            meta.logits(0).map(|_| &mut [][..])
                        };
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
                        barrier.wait();
                    }
                }))
            })
            .collect::<Vec<_>>();

        let sample = RandomSample::new(&Cpu);
        let indices = RandomSample::build_indices(model.meta.nvoc, &ThisThread);
        test_utils::test_infer(eos, tokenizer, &prompt, max_steps, |input, pos| {
            let mut embd = model.meta.embd(input.len()).map(Blob::new);
            let mut logits = model.meta.logits(1).map(Blob::new);

            let d = embd.get().len() / input.len();
            for (i, &tok) in input.iter().enumerate() {
                embd.get_mut()[i * d..][..d]
                    .copy_from_slice(&model.token_embd[tok as usize * d..][..d]);
            }
            let embd = embd.take();

            let barrier = Arc::new(Barrier::new(senders.len() + 1));
            for sender in &senders {
                sender
                    .send(Task {
                        nt: input.len(),
                        pos,
                        embd: embd.as_ptr(),
                        logits: logits.get_mut().as_mut_ptr(),
                        barrier: barrier.clone(),
                    })
                    .unwrap();
            }
            barrier.wait();

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
                    &ThisThread,
                )
                .unwrap();

            pair.idx() as _
        });

        drop(senders);
    })
}

struct Task {
    nt: usize,
    pos: usize,
    embd: *const u8,
    logits: *mut u8,
    barrier: Arc<Barrier>,
}

unsafe impl Send for Task {}

struct WorkerSeed {
    tasks: Receiver<Task>,
    node: InprocNode<usize>,
}

impl WorkerSeed {
    fn new(n: usize) -> (Vec<Self>, Vec<Sender<Task>>) {
        let mut tasks = Vec::with_capacity(n);
        let mut senders = Vec::with_capacity(n);
        let nodes = InprocNode::new(n);
        for _ in 0..n {
            let (sender, receiver) = std::sync::mpsc::channel();
            tasks.push(receiver);
            senders.push(sender);
        }
        (
            zip(nodes, tasks)
                .map(|(node, tasks)| Self { node, tasks })
                .collect(),
            senders,
        )
    }
}
