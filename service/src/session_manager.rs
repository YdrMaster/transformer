use crate::Session;
use causal_lm::CausalLM;
use log::warn;
use lru::LruCache;
use std::{fmt::Debug, hash::Hash, num::NonZeroUsize, sync::Mutex};

pub struct SessionManager<SessionId, M: CausalLM> {
    pending: Mutex<LruCache<SessionId, Option<Session<M>>>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SessionError {
    Busy,
    Duplicate,
    NotFound,
}

impl<SessionId: Eq + Hash + Debug, M: CausalLM> SessionManager<SessionId, M> {
    pub fn new(capacity: Option<usize>) -> Self {
        let cache = capacity
            .map(|c| NonZeroUsize::new(c).expect("Session capacity must be non-zero"))
            .map(LruCache::new)
            .unwrap_or_else(LruCache::unbounded);
        Self {
            pending: Mutex::new(cache),
        }
    }

    pub fn take(&self, k: &SessionId) -> Result<Session<M>, SessionError> {
        self.pending
            .lock()
            .unwrap()
            .get_mut(k)
            .ok_or(SessionError::NotFound)?
            .take()
            .ok_or(SessionError::Busy)
    }

    pub fn get_or_insert(
        &self,
        session_id: SessionId,
        f: impl FnOnce() -> Session<M>,
    ) -> Result<Session<M>, SessionError> {
        self.pending
            .lock()
            .unwrap()
            .get_or_insert_mut(session_id, || Some(f()))
            .take()
            .ok_or(SessionError::Busy)
    }

    pub fn drop_(&self, session_id: &SessionId) -> Result<(), SessionError> {
        if self.pending.lock().unwrap().pop(session_id).is_some() {
            Ok(())
        } else {
            Err(SessionError::NotFound)
        }
    }

    pub fn fork(
        &self,
        session_id: SessionId,
        new_session_id: SessionId,
    ) -> Result<(), SessionError> {
        let mut sessions = self.pending.lock().unwrap();

        if !sessions.contains(&new_session_id) {
            let new = sessions
                .get_mut(&session_id)
                .ok_or(SessionError::NotFound)?
                .as_ref()
                .ok_or(SessionError::Busy)?
                .fork();
            if let Some((out, _)) = sessions.push(new_session_id, Some(new)) {
                warn!("{out:?} dropped because LRU cache is full");
            }
            Ok(())
        } else {
            Err(SessionError::Duplicate)
        }
    }

    pub fn restore(&self, session_id: &SessionId, session: Session<M>) {
        if let Some(option) = self.pending.lock().unwrap().get_mut(session_id) {
            assert!(option.replace(session).is_none());
        }
    }
}
