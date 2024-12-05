use crate::{Operators, RandomSample, Weights};
use gguf::GGufModel;
use llama::{ext::ggml_quants::f16, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use operators::{
    all_reduce::nccl::Operator as AllReduce,
    cuda::{self, memcpy_d2h, NoDevice},
    nccl::CommunicatorGroup,
    nvidia_gpu::NcclNode,
    random_sample::{KVPair, SampleArgs},
    Blob, TopoNode,
};
use regex::Regex;
use std::{
    iter::zip,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::mpsc::{Receiver, Sender},
    thread, usize,
};
use test_utils::{Inference, TokenizerAndPrompt};

type Worker<'w> = LlamaWorker<Operators<NcclNode, AllReduce>, Weights<'w>>;

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

    let devices = match devices {
        Some(devices) => Regex::new(r"\d+")
            .unwrap()
            .find_iter(&devices)
            .map(|c| c.as_str().parse().unwrap())
            .collect::<Vec<_>>(),
        None => vec![0],
    };
    println!("distribution: {devices:?}");

    let lens = vec![1; devices.len()];
    let count = devices.len();

    let (seeds, senders) = match cuda::init() {
        Ok(()) => WorkerSeed::new(&devices),
        Err(NoDevice) => return,
    };
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
                    node.processor().apply(|ctx| {
                        let stream = ctx.stream();
                        let weights = Weights::new(model, range, count, usize::MAX, ctx);
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
                            let mut embd = meta.embd(nt).map(|size| {
                                stream.from_host(unsafe { from_raw_parts(embd, size) })
                            });
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
                                memcpy_d2h(
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
                    });
                }))
            })
            .collect::<Vec<_>>();

        let (next, next_recv) = std::sync::mpsc::channel();
        test_utils::test_infer(eos, tokenizer, &prompt, max_steps, |input, pos| {
            let mut embd = model.meta.embd(input.len()).map(Blob::new);

            let d = embd.get().len() / input.len();
            for (i, &tok) in input.iter().enumerate() {
                embd.get_mut()[i * d..][..d]
                    .copy_from_slice(&model.token_embd[tok as usize * d..][..d]);
            }
            let embd = embd.take();

            for sender in &senders {
                sender
                    .send(Task {
                        nt: input.len(),
                        pos,
                        embd: embd.as_ptr(),
                        next: next.clone(),
                    })
                    .unwrap();
            }
            next_recv.recv().unwrap()
        });

        drop(senders)
    })
}

struct Task {
    nt: usize,
    pos: usize,
    embd: *const u8,
    next: Sender<u32>,
}

unsafe impl Send for Task {}

struct WorkerSeed {
    tasks: Receiver<Task>,
    node: NcclNode,
}

impl WorkerSeed {
    fn new(devices: &[i32]) -> (Vec<Self>, Vec<Sender<Task>>) {
        let nodes = CommunicatorGroup::new(devices)
            .into_vec()
            .into_iter()
            .map(|comm| NcclNode::new(comm, Default::default()))
            .collect::<Vec<_>>();
        let n = nodes.len();
        let mut tasks = Vec::with_capacity(n);
        let mut senders = Vec::with_capacity(n);
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
