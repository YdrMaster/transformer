use operators::Hardware;
use tensor::Tensor;

pub struct Args<'a, H: Hardware> {
    /// shape: [nt, d]
    pub embd: Tensor<&'a mut [H::Byte]>,
    /// shape: [nout, nvoc]
    pub logits: Tensor<&'a mut [H::Byte]>,
    /// shape: [2, _, dh]
    pub sin_cos: Tensor<&'a [H::Byte]>,

    pub requests: Vec<Request<'a, H>>,

    pub num_tokens: usize,
    pub max_seq_len: usize,
    pub max_att_len: usize,
}

pub struct Request<'a, H: Hardware> {
    /// shape: [buf, nblk, nh, dkv+dr]
    pub cache: Tensor<&'a mut [H::Byte]>,
    pub seq_len: usize,
    pub out_len: usize,
    pub pos: usize,
}
