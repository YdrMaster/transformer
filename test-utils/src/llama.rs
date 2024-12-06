use crate::test_infer;
use gguf::{ext::utok, Tokenizer};
use llama::LlamaStorage;
use std::{iter::zip, sync::mpsc};

pub fn test_infer_paralle(
    model: &LlamaStorage<&[u8]>,
    senders: Box<[mpsc::Sender<Task>]>,
    eos: utok,
    tokenizer: Tokenizer,
    prompt: &str,
    max_steps: usize,
) {
    use tensor::Blob;

    let (next, next_recv) = mpsc::channel();
    test_infer(eos, tokenizer, prompt, max_steps, |input, pos| {
        let mut embd = model.meta.embd(input.len()).map(Blob::new).take();

        let d = embd.len() / input.len();
        for (i, &tok) in input.iter().enumerate() {
            embd[i * d..][..d].copy_from_slice(&model.token_embd[tok as usize * d..][..d]);
        }

        for sender in &senders {
            sender
                .send(Task {
                    nt: input.len(),
                    pos,
                    embd: embd.as_ptr(),
                    next: next.clone(),
                })
                .unwrap()
        }
        next_recv.recv().unwrap()
    });
}

pub struct Task {
    pub nt: usize,
    pub pos: usize,
    pub embd: *const u8,
    pub next: mpsc::Sender<utok>,
}

unsafe impl Send for Task {}

pub struct WorkerSeed<N> {
    pub tasks: mpsc::Receiver<Task>,
    pub node: N,
}

impl<N> WorkerSeed<N> {
    pub fn new(nodes: Vec<N>) -> (Vec<Self>, Vec<mpsc::Sender<Task>>) {
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
