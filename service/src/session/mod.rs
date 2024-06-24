mod batcher;
mod cache;
mod dialog;
mod dispatch;
mod task;

use crate::ServiceComponent;
use cache::Cache;
use causal_lm::{CausalLM, SampleArgs};
use dialog::Dialog;
use dispatch::TaskHandle;
use log::info;
use std::{
    cmp::Ordering::{Equal, Greater, Less},
    error, fmt,
    sync::Arc,
    vec,
};

pub(crate) use dispatch::Dispatcher;

/// 会话。
pub struct Session<M: CausalLM> {
    component: Arc<ServiceComponent<M>>,
    pub sample: SampleArgs,

    dialog: Dialog,
    cache: Option<Cache<M::Storage>>,
}

/// 对话错误类型。
///
/// 目前唯一可能的对话错误是增量对话中句子位置异常。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChatError;

impl error::Error for ChatError {}
impl fmt::Display for ChatError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "chat error")
    }
}

impl<M: CausalLM> From<Arc<ServiceComponent<M>>> for Session<M> {
    #[inline]
    fn from(component: Arc<ServiceComponent<M>>) -> Self {
        Self {
            component,
            sample: Default::default(),

            dialog: Default::default(),
            cache: Default::default(),
        }
    }
}

impl<M: CausalLM> Session<M> {
    #[inline]
    pub fn dialog_pos(&self) -> usize {
        self.dialog.num_sentences()
    }

    /// 复制当前会话。
    pub fn fork(&self) -> Self {
        Self {
            component: self.component.clone(),
            sample: self.sample.clone(),
            dialog: self.dialog.clone(),
            cache: self
                .cache
                .as_ref()
                .map(|cache| cache.duplicate(&self.component.handle.model)),
        }
    }

    /// 回滚对话到第 `dialog_pos` 个句子。
    pub fn revert(&mut self, dialog_pos: usize) -> Result<(), ChatError> {
        match dialog_pos.cmp(&self.dialog.num_sentences()) {
            Less => {
                let cache = self.cache.as_mut().unwrap();

                self.dialog.revert(dialog_pos);
                let last_prompt = self.dialog.last_prompt().map_or(0, |p| p.len());
                if cache.revert(self.dialog.num_tokens()).is_none()
                    || cache.get_last_cached_range_len() < last_prompt
                {
                    let len = self.component.handle.model.max_seq_len() as usize;
                    let (tokens, pos) = self.dialog.window(len);
                    cache.reset_with(tokens, pos);
                }
                Ok(())
            }
            Equal => Ok(()),
            Greater => Err(ChatError),
        }
    }

    /// 用 dialog 填充会话。
    pub fn extend<'a>(&mut self, dialog: impl IntoIterator<Item = &'a str>) {
        let eos = self.component.handle.model.eos_token();
        let cache = self
            .cache
            .get_or_insert_with(|| Cache::new(&self.component.handle.model, vec![]));
        // 填充对话
        for s in dialog {
            let prompt = self.dialog.num_sentences() % 2 == 0;

            let s = if prompt {
                self.component.template.apply_chat(s)
            } else {
                s.into()
            };
            let s = self.component.normalizer.encode(&s);
            let mut s = self.component.tokenizer.encode(&s);
            if !prompt {
                s.push(eos);
            }

            cache.extend(&s);
            self.dialog.push(s);
            assert_eq!(cache.end(), self.dialog.num_tokens());
        }
    }

    /// 启动推理任务，返回忙会话。
    pub fn chat(&mut self) -> BusySession<M> {
        let sample = self.sample.clone();
        let cache = self.cache.take().unwrap();
        let handle = self.component.infer(sample, cache);
        BusySession {
            session: self,
            handle,
        }
    }

    fn restore_cache(&mut self, mut cache: Cache<M::Storage>) {
        let end = self.dialog.num_tokens();
        if cache.end() > end {
            // 无论忙会话为何丢弃，只要生成了新句子，就补充一个结束符
            cache.push(self.component.handle.model.eos_token());
            // 只要忙会话收集到任何 token，就生成一个新的句子
            self.dialog.push(cache.slice_tail(end).to_vec());
        }
        cache.cleanup_before_start();
        info!("Cache restored at {} tokens", cache.end());
        self.cache = Some(cache);
    }
}

/// 忙会话，表示会话正在处理推理任务，并可接收推理结果。
pub struct BusySession<'a, M: CausalLM> {
    session: &'a mut Session<M>,
    handle: TaskHandle<M>,
}

impl<M: CausalLM> BusySession<'_, M> {
    /// 接收模型解码产生的文本。
    #[inline]
    pub async fn decode(&mut self) -> Option<String> {
        self.session.component.decode(&mut self.handle).await
    }
}

impl<M: CausalLM> Drop for BusySession<'_, M> {
    #[inline]
    fn drop(&mut self) {
        self.session.restore_cache(self.handle.take());
    }
}

/// 用于文本生成任务的生成器。
pub struct Generator<M: CausalLM> {
    component: Arc<ServiceComponent<M>>,
    handle: TaskHandle<M>,
}

impl<M: CausalLM> Generator<M> {
    pub(crate) fn new(
        component: Arc<ServiceComponent<M>>,
        prompt: impl AsRef<str>,
        sample: SampleArgs,
    ) -> Self {
        let prompt = component.template.normalize(prompt.as_ref());
        let prompt = component.normalizer.encode(&prompt);
        let tokens = component.tokenizer.encode(&prompt);
        let handle = component.infer(sample, Cache::new(&component.handle.model, tokens));
        Self { handle, component }
    }

    /// 接收模型解码产生的文本。
    #[inline]
    pub async fn decode(&mut self) -> Option<String> {
        self.component.decode(&mut self.handle).await
    }
}

impl<M: CausalLM> Drop for Generator<M> {
    #[inline]
    fn drop(&mut self) {
        let _ = self.handle.take();
    }
}
