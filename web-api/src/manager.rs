use crate::schemas;
use actix_web::web;
use service::{Service, Session, SessionHandle};
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{Arc, Mutex},
};

pub struct ServiceManager {
    /// All sessions, session id as key.
    /// New session will be created when a new id comes.
    /// The value will become empty when that session is being served,
    /// so that a new request with the same id will not be double-served.
    /// A session must be re-inserted after being served.
    sessions: Arc<Mutex<HashMap<String, Option<Session>>>>,

    /// The abort handle for all existing sessions, session id as key.
    session_handles: Arc<Mutex<HashMap<String, SessionHandle>>>,

    /// Inference service provided by backend model
    infer_service: Arc<Service>,
}

impl ServiceManager {
    pub fn from(infer_service: Arc<Service>) -> Self {
        ServiceManager {
            sessions: Default::default(),
            session_handles: Default::default(),
            infer_service: infer_service.clone(),
        }
    }
}

impl ServiceManager {
    /// Get existing or create new infer session for a infer request.
    /// Return session or error.
    pub fn get_session(
        &self,
        request: &web::Json<schemas::InferRequest>,
    ) -> Result<Session, schemas::Error> {
        match self
            .sessions
            .lock()
            .unwrap()
            .entry(request.session_id.to_string())
        {
            // Case session id exists
            Entry::Occupied(mut e) => match e.get_mut().take() {
                // If session available, check request
                Some(s) => {
                    if request.first_request {
                        // Session id exists but user thinks otherwise, overwrite current session
                        Ok(
                            e.insert(Some(self.create_session(request.session_id.clone())))
                                .take()
                                .unwrap(),
                        )
                    } else {
                        // take the existing session
                        Ok(s)
                    }
                }
                // If session is being served
                None => Err(schemas::Error::SessionBusy),
            },
            // Case new session id
            Entry::Vacant(e) => {
                if request.first_request {
                    // First request, create new session
                    Ok(
                        e.insert(Some(self.create_session(request.session_id.clone())))
                            .take()
                            .unwrap(),
                    )
                } else {
                    // Session id does not exist but user thinks otherwise, histroy lost
                    Err(schemas::Error::SessionNotFound)
                }
            }
        }
    }

    /// Create a new infer session
    fn create_session(&self, session_id: String) -> Session {
        // launch new infer session
        let session = self.infer_service.launch();
        // register session abort handle
        self.session_handles
            .lock()
            .unwrap()
            .insert(session_id, session.handle());
        return session;
    }

    /// Return the taken-away session, should be called every time a request is done
    pub fn reset_session(&self, id: &str, session: Session) {
        self.sessions
            .lock()
            .unwrap()
            .insert(id.to_string(), Some(session));
    }

    /// Signal the backend model to stop the current inferrence task given session id
    pub fn cancel_session(
        &self,
        request: &web::Json<schemas::CancelRequest>,
    ) -> Result<schemas::Success, schemas::Error> {
        self.session_handles
            .lock()
            .unwrap()
            .get(&request.session_id)
            .ok_or(schemas::Error::CancelFailed)
            .map(|handle| {
                handle.abort();
                schemas::Success::SessionCanceled
            })
    }
}
