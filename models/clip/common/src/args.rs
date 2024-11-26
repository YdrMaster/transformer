use operators::Hardware;
use tensor::Tensor;

pub struct Args<'a, H: Hardware> {
    /// shape: [n, c, h, w]
    pub raw: Tensor<&'a [H::Byte]>,
}
