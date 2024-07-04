use hyper::StatusCode;
use service::SessionError;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(serde::Deserialize)]
pub(crate) struct Infer {
    pub inputs: Vec<Sentence>,
    pub session_id: Option<String>,
    pub dialog_pos: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

#[derive(serde::Deserialize)]
pub(crate) struct Sentence {
    #[allow(unused)]
    pub role: String,
    pub content: String,
}

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct AnonymousSessionId(usize);

impl AnonymousSessionId {
    pub(crate) fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum SessionId {
    Permanent(String),
    Temporary(AnonymousSessionId),
}

#[derive(serde::Deserialize)]
pub struct Fork {
    pub session_id: String,
    pub new_session_id: String,
}

#[derive(serde::Deserialize)]
pub struct Drop_ {
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
pub enum Error {
    Session(SessionError),
    WrongJson(serde_json::Error),
    InvalidDialogPos(usize),
}

#[derive(serde::Serialize)]
struct ErrorBody {
    status: u16,
    code: u16,
    message: String,
}

impl Error {
    #[inline]
    pub const fn status(&self) -> StatusCode {
        use SessionError::*;
        match self {
            Self::Session(NotFound) => StatusCode::NOT_FOUND,
            Self::Session(Busy) => StatusCode::NOT_ACCEPTABLE,
            Self::Session(Duplicate) => StatusCode::CONFLICT,
            Self::WrongJson(_) => StatusCode::BAD_REQUEST,
            Self::InvalidDialogPos(_) => StatusCode::RANGE_NOT_SATISFIABLE,
        }
    }

    #[inline]
    pub fn body(&self) -> serde_json::Value {
        macro_rules! error {
            ($code:expr, $msg:expr) => {
                ErrorBody {
                    status: self.status().as_u16(),
                    code: $code,
                    message: $msg.into(),
                }
            };
        }

        #[inline]
        fn json(v: impl serde::Serialize) -> serde_json::Value {
            serde_json::to_value(v).unwrap()
        }

        use SessionError::*;
        match self {
            Self::Session(NotFound) => json(error!(0, "Session not found")),
            Self::Session(Busy) => json(error!(0, "Session is busy")),
            Self::Session(Duplicate) => json(error!(0, "Session ID already exists")),
            Self::WrongJson(e) => json(error!(0, e.to_string())),
            &Self::InvalidDialogPos(current_dialog_pos) => {
                #[derive(serde::Serialize)]
                struct ErrorBodyExtra {
                    #[serde(flatten)]
                    common: ErrorBody,
                    current_dialog_pos: usize,
                }
                json(ErrorBodyExtra {
                    common: error!(0, "Dialog position out of range"),
                    current_dialog_pos,
                })
            }
        }
    }
}
