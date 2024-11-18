use super::GGufModel;
use ggus::{GGmlTokenType, GGufMetaMapExt};
use std::{
    borrow::Cow,
    str::{from_utf8, from_utf8_unchecked},
};
use tokeneer::{utok, Bpe, Lpe, Method, Tokeneer};

pub struct Tokenizer {
    tokenize: Box<dyn Tokenize>,
    replace_space: Option<char>,
}

impl GGufModel<'_> {
    pub fn tokenizer(&self) -> Tokenizer {
        match self.tokenizer_ggml_model().unwrap() {
            "llama" => Tokenizer::bpe_from_gguf(self),
            "fm9g8b" => Tokenizer::lpe_from_gguf(self),
            model => panic!("Unsupported tokenizer model: {model}"),
        }
    }
}

impl Tokenizer {
    pub fn encode(&self, text: &str) -> Vec<utok> {
        let space = self.replace_space.unwrap_or(' ');
        let mut chars = text.chars();
        let mut text = match chars.next() {
            Some(c) => {
                if c.is_ascii_alphabetic() {
                    format!("{space}{c}")
                } else {
                    format!("{c}")
                }
            }
            None => return vec![],
        };
        for c in chars {
            text.push(match c {
                ' ' => space,
                c => c,
            })
        }
        self.tokenize.encode(&text)
    }
    pub fn decode(&self, token: utok) -> Cow<str> {
        let piece = self.tokenize.decode(token);
        if let Some(c) = self.replace_space {
            piece.replace(c, " ").into()
        } else {
            piece.into()
        }
    }

    fn bpe_from_gguf(gguf: &GGufModel) -> Self {
        let _pre = gguf.get_str("tokenizer.ggml.pre").unwrap();
        let tokens = gguf.tokenizer_ggml_tokens().unwrap();
        let scores = gguf.tokenizer_ggml_scores().unwrap();
        let token_type = gguf.tokenizer_ggml_token_type().unwrap();
        assert_eq!(tokens.len(), scores.len());
        assert_eq!(tokens.len(), token_type.len());

        let mut space_exist = false;
        let mut replace_exist = false;
        let vocabs = tokens.map(|piece| {
            let piece = piece.unwrap();
            match piece {
                " " => space_exist = true,
                "▁" => replace_exist = true,
                _ => {}
            }
            piece
        });
        let scores = scores.map(|score| score.unwrap());
        let is_byte = token_type.map(|ty| GGmlTokenType::Byte as i32 == ty.unwrap());

        let unk = gguf.tokenizer_ggml_unknown_token_id().unwrap();
        let bos = gguf.tokenizer_ggml_bos_token_id().unwrap();
        let eos = gguf.tokenizer_ggml_eos_token_id().unwrap();

        let bpe = Bpe::new(vocabs, scores, is_byte, unk);
        let bos_piece = from_utf8(bpe.decode(bos)).unwrap().to_string();
        let eos_piece = from_utf8(bpe.decode(eos)).unwrap().to_string();

        let mut tokeneer = Tokeneer::new(bpe);
        tokeneer.extend_special([(bos_piece, vec![bos]), (eos_piece, vec![eos])]);
        Self {
            tokenize: Box::new(tokeneer),
            replace_space: match (space_exist, replace_exist) {
                (_, true) => Some('▁'),
                (true, false) => None,
                (false, false) => panic!("Unknown user-defined space"),
            },
        }
    }

    fn lpe_from_gguf(gguf: &GGufModel) -> Self {
        let tokens = gguf.tokenizer_ggml_tokens().unwrap();

        let mut space_exist = false;
        let mut replace_exist = false;
        let vocabs = tokens.map(|piece| {
            let piece = piece.unwrap();
            match piece {
                " " => space_exist = true,
                "▁" => replace_exist = true,
                _ => {}
            }
            piece.as_bytes()
        });

        let unk = gguf.tokenizer_ggml_unknown_token_id().unwrap();
        let bos = gguf.tokenizer_ggml_bos_token_id().unwrap();
        let eos = gguf.tokenizer_ggml_eos_token_id().unwrap();

        let bpe = Lpe::new(vocabs, unk);
        let bos_piece = from_utf8(bpe.decode(bos)).unwrap().to_string();
        let eos_piece = from_utf8(bpe.decode(eos)).unwrap().to_string();

        let mut tokeneer = Tokeneer::new(bpe);
        tokeneer.extend_special([(bos_piece, vec![bos]), (eos_piece, vec![eos])]);
        Self {
            tokenize: Box::new(tokeneer),
            replace_space: match (space_exist, replace_exist) {
                (_, true) => Some('▁'),
                (true, false) => None,
                (false, false) => panic!("Unknown user-defined space"),
            },
        }
    }
}

/// A trait for tokenization.
trait Tokenize {
    /// Encode a text into a sequence of tokens.
    fn encode(&self, text: &str) -> Vec<utok>;
    /// Decode a token into str.
    fn decode(&self, token: utok) -> &str;
}

impl<M: tokeneer::Method> Tokenize for Tokeneer<M> {
    #[inline]
    fn encode(&self, text: &str) -> Vec<utok> {
        self.encode(text)
    }
    #[inline]
    fn decode(&self, token: utok) -> &str {
        unsafe { from_utf8_unchecked(self.internal().decode(token)) }
    }
}

#[test]
fn test_load() {
    use test_utils::Inference;
    let Some(Inference { model, prompt, .. }) = Inference::load() else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));
    let tokenizer = gguf.tokenizer();
    println!("{:?}", tokenizer.encode(&prompt));
}
