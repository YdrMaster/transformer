use causal_lm::{CausalLM, QueryContext};
use common::{upos, utok};
use log::{debug, info};
use rangemap::{range_set, RangeSet};
use std::{cmp::min, ops::Range};
use tensor::Tensor;

pub(super) struct Cache<Storage> {
    /// 可映射的 token 序列。
    tokens: Vec<utok>,
    /// token 序列在整个对话中的位置。
    pos: usize,
    /// 需要缓存的 token 在 token 序列中的范围。
    cached: RangeSet<usize>,
    /// 已缓存的 token 在 cached_range 中的范围
    to_be_cached: RangeSet<usize>,
    /// 计算缓存。
    cache: Tensor<Storage>,
}

pub struct CacheQuery<'a> {
    tokens: &'a [utok],
    to_be_cached: &'a RangeSet<usize>,
}

impl<'a> CacheQuery<'a> {
    fn new(tokens: &'a [utok], to_be_cached: &'a RangeSet<usize>) -> Self {
        Self {
            tokens,
            to_be_cached,
        }
    }

    pub fn len(&self) -> usize {
        self.to_be_cached.iter().map(|range| range.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> IntoIterator for CacheQuery<'a> {
    type Item = &'a utok;

    type IntoIter = CacheQueryIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        CacheQueryIter::new(self.tokens, self.to_be_cached)
    }
}

pub struct CacheQueryIter<'a> {
    tokens: &'a [utok],
    ranges: rangemap::set::Iter<'a, usize>,
    current_iter: Option<Range<usize>>,
}

impl<'a> CacheQueryIter<'a> {
    fn new(tokens: &'a [utok], to_be_cached: &'a RangeSet<usize>) -> Self {
        let mut ranges_iter = to_be_cached.iter();

        let current_iter = ranges_iter.next().cloned();

        Self {
            tokens,
            ranges: ranges_iter,
            current_iter,
        }
    }
}

impl<'a> Iterator for CacheQueryIter<'a> {
    type Item = &'a utok;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(range) = &mut self.current_iter {
            if let Some(i) = range.next() {
                Some(&self.tokens[i])
            } else {
                self.current_iter = self.ranges.next().cloned();
                self.current_iter.as_mut()?.next().map(|i| &self.tokens[i])
            }
        } else {
            None
        }
    }
}

impl<Storage> Cache<Storage> {
    /// 生成一个空白的缓存结构，准备填充 `tokens`。
    #[inline]
    pub fn new(t: &impl CausalLM<Storage = Storage>, tokens: Vec<utok>) -> Self {
        let tokens_len = tokens.len();
        Self {
            tokens,
            pos: 0,
            cached: RangeSet::new(),
            to_be_cached: range_set![0..tokens_len],
            cache: t.new_cache(),
        }
    }

    /// 复制缓存结构。
    #[inline]
    pub fn duplicate(&self, t: &impl CausalLM<Storage = Storage>) -> Self {
        debug!("call duplicate");
        Self {
            tokens: self.tokens.clone(),
            pos: self.pos,
            cached: self.cached.clone(),
            to_be_cached: self.to_be_cached.clone(),
            cache: t.duplicate_cache(&self.cache, self.cached_len() as _),
        }
    }
    /// 回滚缓存到 `pos`，并返回剩余的有效缓存长度。
    pub fn revert(&mut self, pos: usize) -> Option<usize> {
        debug!("call revert");
        // 回滚之后，tokens.len()、cached.end、pos 不能大于新的 pos
        // 1. pos 不大于 pos；
        let len = pos.checked_sub(self.pos)?;
        // 2. cached.end 不大于 pos；
        if len != 0 && self.cached.contains(&(len - 1)) {
            self.to_be_cached.clear();
            self.cached.remove(len..self.cached.last().unwrap().end);
        } else {
            return None;
        }
        // 3. tokens.len() 不大于 pos；
        self.tokens.truncate(len);
        // 返回当前的缓存长度
        Some(self.cached_len())
    }
    /// 扩展待填充 token。
    #[inline]
    pub fn extend(&mut self, tokens: &[utok]) {
        debug!("call extend");
        let before_len = self.tokens.len();
        self.tokens.extend_from_slice(tokens);
        self.to_be_cached.insert(before_len..self.tokens.len());
    }
    /// 所有 token 中还没有加入缓存的部分就是这次的查询。
    #[inline]
    pub fn query(&self) -> CacheQuery {
        CacheQuery::new(&self.tokens, &self.to_be_cached)
    }
    /// 生成对应的查询上下文。
    #[inline]
    pub fn as_ctx(&mut self) -> QueryContext<Storage> {
        debug!("call as_ctx");
        debug!(
            "cache reset\ncached is {:?}\nto_be_cached is {:?}",
            self.cached, self.to_be_cached
        );
        QueryContext {
            range: self.cached_len() as upos..(self.cached_len() + self.to_be_cached_len()) as upos,
            cache: Some(&mut (self.cache)),
        }
    }

    /// 将新采样的值加入缓存。默认to_be_cached不为空
    #[inline]
    pub fn push(&mut self, token: utok) {
        debug!("call push");
        assert!(self.is_continue());

        //to_be_cached 全部变为cached
        self.to_be_cached
            .iter()
            .for_each(|range| self.cached.insert(range.clone()));
        //清空to_be_cached 并插入新的需要缓存的token
        self.to_be_cached.clear();
        self.to_be_cached
            .insert(self.tokens.len()..self.tokens.len() + 1);
        //插入token
        self.tokens.push(token);
    }
    /// 已采样的最后一个词在对话中的位置。
    #[inline]
    pub fn end(&self) -> usize {
        self.pos + self.tokens.len()
    }
    /// 提取尾部词序列，默认尾部序列在cached和to_be_cached中是连续的
    #[inline]
    pub fn slice_tail(&self, pos: usize) -> &[utok] {
        let known = pos.checked_sub(self.pos).unwrap();
        &self.tokens[known..]
    }

    /// 重置缓存窗口,并将起始点设置为尾部一部分之前
    #[allow(unused)]
    pub fn reset_within_one_range(&mut self, min: usize, max: usize) {
        if self.cached_len() + self.to_be_cached_len() >= max {
            self.cached.clear();
            self.to_be_cached = range_set![(self.tokens.len() - min..self.tokens.len())];
        }
    }
    /// 重置缓存窗口，保留起始的一部分，并将起始点设置为尾部一部分之前
    pub fn reset_within_start_and_end_range(
        &mut self,
        start_size: usize,
        end_size: usize,
        max: usize,
    ) {
        assert!(start_size + end_size <= max);
        if self.cached_len() + self.to_be_cached_len() >= max {
            let mut uncached_start: usize = 0;

            self.cached.clear();
            if let Some(mut first_range) = self.cached.first().cloned() {
                first_range.end = min(first_range.end, start_size);
                if first_range.start == 0 && !first_range.is_empty() {
                    uncached_start = first_range.end;
                    self.cached = range_set![first_range];
                }
            }
            // 为to_be_cached 赋值
            if uncached_start != start_size {
                self.to_be_cached = range_set![
                    uncached_start..start_size,
                    self.tokens.len() - end_size..self.tokens.len()
                ];
            } else {
                self.to_be_cached = range_set![self.tokens.len() - end_size..self.tokens.len()];
            }

            info!(
                "cache reset\ncached is {:?}\nto_be_cached is {:?}",
                self.cached, self.to_be_cached
            );
        }
    }
    /// 重置并清空缓存窗口。
    pub fn reset_with(&mut self, tokens: Vec<utok>, pos: usize) {
        self.tokens = tokens;
        self.pos = pos;
        self.cached.clear();
        self.to_be_cached = range_set![0..self.tokens.len()];
    }
    /// 清理缓存中在缓存窗口之前的部分。
    pub fn cleanup_before_start(&mut self) {
        let to_remove = self.cached.first().unwrap().start;
        if to_remove > 0 {
            self.tokens.copy_within(to_remove.., 0);
            self.pos += to_remove;
            self.tokens.truncate(self.tokens.len() - to_remove);

            // 整体减小cached和to_be_cached
            self.cached
                .iter()
                .fold(RangeSet::new(), |mut set, range| {
                    set.insert(range.start - to_remove..range.end - to_remove);
                    set
                })
                .clone_into(&mut self.cached);
            self.to_be_cached
                .iter()
                .fold(RangeSet::new(), |mut set, range| {
                    set.insert(range.start - to_remove..range.end - to_remove);
                    set
                })
                .clone_into(&mut self.to_be_cached);
        }
    }
    /// 获取cached中最后一个区间的长度，如果cached为空则会panic
    pub fn get_last_cached_range_len(&self) -> usize {
        self.cached.last().unwrap().len()
    }

    /// 判定需要缓存的部分包含tokens的结尾
    fn is_continue(&self) -> bool {
        if self.to_be_cached.is_empty() {
            self.cached.last().unwrap().end == self.tokens.len()
        } else {
            self.to_be_cached.last().unwrap().end == self.tokens.len()
        }
    }

    /// 获取cached 总长度
    #[inline]
    fn cached_len(&self) -> usize {
        self.cached.iter().map(|range| range.len()).sum()
    }

    /// 获取 to_be_cached 总长度
    #[inline]
    fn to_be_cached_len(&self) -> usize {
        self.to_be_cached.iter().map(|range| range.len()).sum()
    }
}

#[test]
fn test_cache_query() {
    let v: Vec<u32> = (0..100).into_iter().collect();
    CacheQuery::new(
        &v,
        &range_set![10 as usize..20 as usize, 40 as usize..50 as usize],
    )
    .into_iter()
    .for_each(|a| println!("{:?}", a));
}
