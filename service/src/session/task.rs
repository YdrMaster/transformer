use super::cache::Cache;
use causal_lm::SampleArgs;
use common::utok;
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::mpsc::UnboundedSender;

pub(super) struct Task<Storage> {
    sample: SampleArgs,
    sender: UnboundedSender<utok>,

    cache: Arc<Mutex<Option<Cache<Storage>>>>,
}

impl<Storage> Task<Storage> {
    #[inline]
    pub fn new(
        cache: Arc<Mutex<Option<Cache<Storage>>>>,
        sample: SampleArgs,
        sender: UnboundedSender<utok>,
    ) -> Self {
        Self {
            sample,
            sender,
            cache,
        }
    }

    #[inline]
    pub fn sample(&self) -> &SampleArgs {
        &self.sample
    }
    #[inline]
    pub fn is_alive(&self) -> bool {
        !self.sender.is_closed()
    }
    #[inline]
    pub fn lock_cache(&self) -> MutexGuard<Option<Cache<Storage>>> {
        self.cache.lock().unwrap()
    }

    #[inline]
    pub fn push(&mut self, token: utok, start_size: usize, end_size: usize, max: usize) -> bool {
        if self.sender.send(token).is_ok() {
            if let Some(cache) = self.cache.lock().unwrap().as_mut() {
                cache.push(token);
                cache.reset_within_start_and_end_range(start_size, end_size, max);
                return true;
            }
        }
        false
    }
}
