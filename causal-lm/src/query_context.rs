use common::upos;
use std::ops::{DerefMut, Range};
use tensor::{slice, split, udim, LocalSplitable, Tensor};

/// 查询 Transformer 的的信息。
pub struct QueryContext<'a, Storage> {
    /// K-V cache.
    pub cache: Option<&'a mut Tensor<Storage>>,
    /// 查询在上下文中的位置。
    pub range: Range<upos>,
}

impl<Storage> QueryContext<'_, Storage> {
    /// 查询的位置。
    #[inline]
    pub const fn pos(&self) -> upos {
        self.range.start
    }
    /// 查询的长度。
    #[inline]
    pub fn seq_len(&self) -> udim {
        self.range.len() as _
    }
    /// 注意力长度。
    #[inline]
    pub const fn att_len(&self) -> udim {
        self.range.end
    }
}

type KVCache<'a, T> = (
    Tensor<LocalSplitable<&'a mut [T]>>,
    Tensor<LocalSplitable<&'a mut [T]>>,
);

impl<Storage, T> QueryContext<'_, Storage>
where
    Storage: DerefMut<Target = [T]>,
{
    /// 提取第 `layer` 层的 K-V 缓存。
    pub fn cache(&mut self, layer: usize) -> Option<KVCache<T>> {
        self.cache.as_mut().map(|cache| {
            let &[_, 2, nkvh, max_seq_len, dh] = cache.shape() else {
                unreachable!()
            };
            let u = cache
                .as_mut()
                .map_physical(|u| LocalSplitable::from(&mut **u))
                .slice(&[
                    slice![=layer],
                    slice![=>],
                    slice![=>],
                    slice![=>],
                    slice![=>],
                ]);
            let (k, v) = split!(u; [1]: 1, 1);
            (
                k.reshape(&[nkvh, max_seq_len, dh]),
                v.reshape(&[nkvh, max_seq_len, dh]),
            )
        })
    }
}
