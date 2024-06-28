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
