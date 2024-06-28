use crate::utils::{
    SessionId,
    Error,
    ForkSuccess,
    DropSuccess
};
use service::Session;
use causal_lm::CausalLM;
use lru::LruCache;
use std::{
        num::NonZeroUsize, sync::{
        Mutex, MutexGuard
    }
};

pub struct SessionManager<M: CausalLM> {
    pending: Mutex<LruCache<SessionId, Option<Session<M>>>>
}

impl<M: CausalLM> SessionManager<M> {
    pub fn new(capacity: Option<usize>) -> Self {
        let cap = capacity.map(|c| NonZeroUsize::new(c).expect("Session capacity must be non-zero"));
        Self {
            pending: Mutex::new(cap.map(LruCache::new).unwrap_or_else(LruCache::unbounded)),
        }
    }

    pub fn get_mut(&self, k: &SessionId) -> Result<Session<M>, Error> {
        self.pending
            .lock()
            .unwrap()
            .get_mut(&k)
            .ok_or(Error::SessionNotFound)?
            .take()
            .ok_or(Error::SessionBusy)
    }

    pub fn get_or_insert_mut(&self, session_id: &SessionId, v: Session<M>) -> Result<Session<M>, Error> {
        self.pending
            .lock()
            .unwrap()
            .get_or_insert_mut(session_id.clone(),
                || { info!("{:?} created", &session_id);
                Some(v)})
            .take()
            .ok_or(Error::SessionBusy)
    }

    pub fn drop(&self, session_id: &SessionId) -> Result<DropSuccess, Error>{
        if self.pending
            .lock()
            .unwrap()
            .pop(&session_id)
            .is_some() {
                info!("{session_id:?} dropped in drop function");
                Ok(DropSuccess)
            } else {
                Err(Error::SessionNotFound)
            }
    }

    pub fn fork(&self, session_id: String, new_session_id: String) -> Result<ForkSuccess, Error> {
        let new_session_id_warped = SessionId::Permanent(new_session_id.clone());
        let mut sessions = self.get_sessions();

        if !sessions.contains(&new_session_id_warped) {
            let new = sessions
                .get_mut(&SessionId::Permanent(session_id.clone()))
                .ok_or(Error::SessionBusy)?
                .as_ref()
                .ok_or(Error::SessionBusy)?
                .fork();
            info!("{new_session_id} is forked from {session_id:?}");
            if let Some((out, _)) = sessions.push(new_session_id_warped, Some(new)) {
                warn!("{out:?} dropped because LRU cache is full");
            }
            Ok(ForkSuccess)
        } else {
            Err(Error::SessionDuplicate)
        }
    }

    pub fn get_sessions(&self) -> MutexGuard<LruCache<SessionId, Option<Session<M>>>> {
        self.pending.lock().unwrap()
    }

    pub fn restore(&self, session_id: &SessionId, session: Session<M>)
    {
        if let Some(option) = self.pending.lock().unwrap().get_mut(session_id) {
            assert!(option.replace(session).is_none());
        }
    }
}
