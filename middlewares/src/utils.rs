#[derive(serde::Deserialize)]
pub struct Fork {
    pub session_id: String,
    pub new_session_id: String,
}

#[derive(serde::Deserialize)]
pub struct Drop {
    pub session_id: String,
}

pub struct ForkSuccess;
pub struct DropSuccess;

pub trait Success {
    fn msg(&self) -> &str;
}

impl Success for ForkSuccess {
    fn msg(&self) -> &str {
        "fork success"
    }
}
impl Success for DropSuccess {
    fn msg(&self) -> &str {
        "drop success"
    }
}

#[derive(Debug)]
pub enum SessionError {
    SessionBusy,
    SessionDuplicate,
    SessionNotFound,
}