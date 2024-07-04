#![deny(warnings)]

mod session;
mod session_manager;
mod template;

use causal_lm::{CausalLM, SampleArgs};
use session::{Dispatcher, Generator};
use std::{fmt::Debug, path::Path, sync::Arc};
use template::Template;
use tokenizer::{BPECommonNormalizer, Normalizer, Tokenizer, VocabTxt, BPE};
use tokio::task::JoinHandle;

pub use session::{BusySession, ChatError, Session};
pub use session_manager::{SessionError, SessionManager};

/// 对话服务。
pub struct Service<M: CausalLM> {
    component: Arc<ServiceComponent<M>>,
    pub default_sample: SampleArgs,
}

/// 服务中不变的组件，将在所有会话之间共享。
///
/// 推理线程的生命周期与这个组件绑定。
struct ServiceComponent<M: CausalLM> {
    handle: Arc<Dispatcher<M>>,
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    normalizer: Box<dyn Normalizer + Send + Sync>,
    template: Box<dyn Template + Send + Sync>,
}

impl<M: CausalLM> Drop for ServiceComponent<M> {
    #[inline]
    fn drop(&mut self) {
        // 停止推理任务
        self.handle.stop();
    }
}

impl<M> Service<M>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
    M::Error: Debug,
{
    pub fn load(model_dir: impl AsRef<Path>, meta: M::Meta) -> (Self, JoinHandle<()>) {
        let handle = Arc::new(Dispatcher::from(M::load(&model_dir, meta).unwrap()));
        (
            Self {
                component: Arc::new(ServiceComponent {
                    handle: handle.clone(),
                    tokenizer: tokenizer(&model_dir),
                    normalizer: normalizer(&model_dir),
                    template: template(model_dir),
                }),
                default_sample: Default::default(),
            },
            tokio::task::spawn_blocking(move || handle.run()),
        )
    }
}

impl<M: CausalLM> Service<M> {
    /// 从对话服务启动一个会话。
    #[inline]
    pub fn launch(&self) -> Session<M> {
        let mut session: Session<M> = self.component.clone().into();
        session.sample = self.default_sample.clone();
        session
    }

    /// 从对话服务启动一个文本生成器。
    #[inline]
    pub fn generate(&self, prompt: impl AsRef<str>, sample: Option<SampleArgs>) -> Generator<M> {
        let sample = sample.unwrap_or_else(|| self.default_sample.clone());
        Generator::new(self.component.clone(), prompt, sample)
    }
}

#[test]
fn test() {
    use colored::{Color, Colorize};
    use std::{io::Write, iter::zip};
    use tokio::{runtime::Builder, task::JoinSet};

    let Some(model_dir) = common::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    let runtime = Builder::new_current_thread().build().unwrap();
    let _rt = runtime.enter();

    let (service, _handle) = Service::<llama_cpu::Transformer>::load(model_dir, ());

    let mut set = JoinSet::new();
    let tasks = vec![
        ("Say \"Hi\" to me.", Color::Yellow),
        ("Hi", Color::Red),
        ("Where is the capital of France?", Color::Green),
    ];

    let sessions = tasks.iter().map(|_| service.launch()).collect::<Vec<_>>();

    for ((prompt, color), mut session) in zip(tasks, sessions) {
        set.spawn(async move {
            session.extend([prompt]);
            let mut busy = session.chat();
            while let Some(s) = busy.decode().await {
                print!("{}", s.color(color));
                std::io::stdout().flush().unwrap();
            }
        });
    }

    runtime.block_on(async { while set.join_next().await.is_some() {} });
    runtime.shutdown_background();
}

fn template(model_dir: impl AsRef<Path>) -> Box<dyn Template + Send + Sync> {
    let path: String = model_dir.as_ref().display().to_string();
    let path = path.to_ascii_lowercase();
    if path.contains("tinyllama") {
        Box::new(template::ChatTinyLlama)
    } else {
        Box::new(template::ChatCPM)
    }
}

fn normalizer(model_dir: impl AsRef<Path>) -> Box<dyn Normalizer + Send + Sync> {
    use std::io::ErrorKind::NotFound;
    match BPE::from_model_file(model_dir.as_ref().join("tokenizer.model")) {
        Ok(_) => return Box::new(BPECommonNormalizer {}),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    match VocabTxt::from_txt_file(model_dir.as_ref().join("vocabs.txt")) {
        Ok(_) => return Box::new(()),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    panic!("Tokenizer file not found");
}

fn tokenizer(model_dir: impl AsRef<Path>) -> Box<dyn Tokenizer + Send + Sync> {
    use std::io::ErrorKind::NotFound;
    match BPE::from_model_file(model_dir.as_ref().join("tokenizer.model")) {
        Ok(bpe) => return Box::new(bpe),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    match VocabTxt::from_txt_file(model_dir.as_ref().join("vocabs.txt")) {
        Ok(voc) => return Box::new(voc),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    panic!("Tokenizer file not found");
}
