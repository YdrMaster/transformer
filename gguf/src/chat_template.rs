use super::{tokenizer::Tokenizer, GGufModel};
use ggus::{GGufMetaError, GGufMetaMapExt};
use minijinja::Environment;
use serde::Serialize;
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    LazyLock, RwLock,
};

/// A template for rendering chat messages.
pub struct ChatTemplate {
    id: String,
    bos: String,
    eos: String,
}

#[derive(Serialize)]
pub struct Message<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

impl GGufModel<'_> {
    pub fn chat_template(&self, tokenizer: &Tokenizer) -> Option<ChatTemplate> {
        let template = match self.tokenizer_chat_template() {
            Ok(t) => t,
            Err(GGufMetaError::NotExist) => return None,
            Err(e) => panic!("Failed to get tokenizer chat template: {e:?}"),
        };
        let bos = self.tokenizer_ggml_bos_token_id().unwrap();
        let eos = self.tokenizer_ggml_eos_token_id().unwrap();

        Some(ChatTemplate::new(
            template.into(),
            tokenizer.decode(bos).into(),
            tokenizer.decode(eos).into(),
        ))
    }
}

impl ChatTemplate {
    /// Create a new chat template.
    pub fn new(template: String, bos: String, eos: String) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, Relaxed).to_string();

        JINJA_ENV
            .write()
            .unwrap()
            .add_template_owned(id.clone(), template)
            .unwrap();

        Self { id, bos, eos }
    }

    /// Render the chat template with the given messages.
    pub fn render(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, minijinja::Error> {
        #[derive(Serialize)]
        struct Args<'a> {
            messages: &'a [Message<'a>],
            bos_token: &'a str,
            eos_token: &'a str,
            add_generation_prompt: bool,
        }

        JINJA_ENV
            .read()
            .unwrap()
            .get_template(&self.id)
            .unwrap()
            .render(Args {
                messages,
                bos_token: &self.bos,
                eos_token: &self.eos,
                add_generation_prompt,
            })
    }
}

impl Drop for ChatTemplate {
    fn drop(&mut self) {
        JINJA_ENV.write().unwrap().remove_template(&self.id);
    }
}

static JINJA_ENV: LazyLock<RwLock<Environment<'_>>> = LazyLock::new(|| {
    let mut env = Environment::new();
    env.set_unknown_method_callback(|_, value, method, args| {
        use minijinja::{value::ValueKind as ThisType, ErrorKind::UnknownMethod, Value};
        match (method, value.kind(), args) {
            ("strip", ThisType::String, []) => Ok(Value::from_safe_string(
                value.to_str().unwrap().trim().into(),
            )),
            _ => Err(UnknownMethod.into()),
        }
    });
    RwLock::new(env)
});

#[test]
fn test() {
    const TAIDE: &str = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = '<<SYS>>\n' + messages[0]['content'] + '\n<</SYS>>\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content + ' [/INST]'}}{% elif message['role'] == 'assistant' %}{{ ' '  + content + ' ' + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
    const MINICPM: &str = "{% for message in messages %}{% if message['role'] == 'user' %}{{'<用户>' + message['content'].strip() + '<AI>'}}{% else %}{{message['content'].strip()}}{% endif %}{% endfor %}";

    let result = ChatTemplate::new(TAIDE.into(), "<s>".into(), "</s>".into())
        .render(
            &[Message {
                role: "user",
                content: "Hello, who are you?",
            }],
            true,
        )
        .unwrap();

    assert_eq!(
        result,
        "<s>[INST] Hello, who are you? [/INST]<|im_start|>assistant\n"
    );

    let result = ChatTemplate::new(MINICPM.into(), "<s>".into(), "</s>".into())
        .render(
            &[Message {
                role: "user",
                content: "Hello, who are you?",
            }],
            true,
        )
        .unwrap();
    assert_eq!(result, "<用户>Hello, who are you?<AI>");
}

#[test]
fn test_load() {
    use test_utils::Inference;
    let Some(Inference { model, .. }) = Inference::load() else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));
    let tokenizer = gguf.tokenizer();
    let chat_template = gguf.chat_template(&tokenizer);
    println!("{:?}", chat_template.map(|t| t.id.clone()));
}
