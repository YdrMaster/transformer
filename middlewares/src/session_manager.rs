use crate::utils::{
    SessionError,
    ForkSuccess,
    DropSuccess
};
use std::{
    fmt::Debug, hash::Hash, num::NonZeroUsize, sync::{
        Mutex, MutexGuard
    }
};
use service::Session;
use causal_lm::CausalLM;
use lru::LruCache;

pub struct SessionManager<SessionId, M: CausalLM> {
    pending: Mutex<LruCache<SessionId, Option<Session<M>>>>
}

impl<SessionId: Clone + Eq + PartialEq + Hash + Debug, M: CausalLM> SessionManager<SessionId, M> {
    pub fn new(capacity: Option<usize>) -> Self {
        let cap = capacity.map(|c| NonZeroUsize::new(c).expect("Session capacity must be non-zero"));
        Self {
            pending: Mutex::new(cap.map(LruCache::new).unwrap_or_else(LruCache::unbounded)),
        }
    }

    pub fn get_mut(&self, k: &SessionId) -> Result<Session<M>, SessionError> {
        self.pending
            .lock()
            .unwrap()
            .get_mut(&k)
            .ok_or(SessionError::SessionNotFound)?
            .take()
            .ok_or(SessionError::SessionBusy)
    }

    pub fn get_or_insert_mut(&self, session_id: &SessionId, v: Session<M>) -> Result<Session<M>, SessionError> {
        self.pending
            .lock()
            .unwrap()
            .get_or_insert_mut(session_id.clone(),
                || { Some(v) })
            .take()
            .ok_or(SessionError::SessionBusy)
    }

    pub fn drop(&self, session_id: &SessionId) -> Result<DropSuccess, SessionError>{
        if self.pending
            .lock()
            .unwrap()
            .pop(&session_id)
            .is_some() {
                Ok(DropSuccess)
            } else {
                Err(SessionError::SessionNotFound)
            }
    }

    pub fn fork(&self, session_id: &SessionId, new_session_id: &SessionId) -> Result<ForkSuccess, SessionError> {
        let mut sessions = self.get_sessions();

        if !sessions.contains(&new_session_id) {
            let new = sessions
                .get_mut(session_id)
                .ok_or(SessionError::SessionNotFound)?
                .as_ref()
                .ok_or(SessionError::SessionBusy)?
                .fork();
            if let Some((out, _)) = sessions.push(new_session_id.clone(), Some(new)) {
                warn!("{out:?} dropped because LRU cache is full");
            }
            Ok(ForkSuccess)
        } else {
            Err(SessionError::SessionDuplicate)
        }
    }

    pub fn restore(&self, session_id: &SessionId, session: Session<M>)
    {
        if let Some(option) = self.pending.lock().unwrap().get_mut(session_id) {
            assert!(option.replace(session).is_none());
        }
    }

    fn get_sessions(&self) -> MutexGuard<LruCache<SessionId, Option<Session<M>>>> {
        self.pending.lock().unwrap()
    }
}
