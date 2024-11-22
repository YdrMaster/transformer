use def::*;
use image::ImageReader;
use itertools::izip;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{iter::zip, ops::Deref, path::Path, slice::from_raw_parts_mut};
use tensor::{rearrange, Blob, Tensor};

#[repr(transparent)]
pub struct Image<T>(Tensor<T>);

pub struct ImageGrid {
    grid: Option<Tensor<Blob>>,
    whole: Image<Blob>,
}

#[allow(non_camel_case_types)]
mod def {
    use gguf::ggml_quants::digit_layout::layout;
    layout!(Urgb u(8)     ; 3);
    layout!(Frgb e(8)m(23); 3);

    /// 有图像尺寸参与的浮点运算应使用这个类型
    pub type fdim = f64;
    /// 基本的 rgb 类型
    pub type urgb = [u8; 3];
    /// 归一化浮点表示的 rgb 类型
    pub type frgb = [f32; 3];
}

impl Image<Vec<u8>> {
    /// 从文件加载
    pub fn load(path: impl AsRef<Path>) -> Self {
        let rgb8 = ImageReader::open(path).unwrap().decode().unwrap().to_rgb8();
        let (x, y) = rgb8.dimensions();
        assert_eq!(rgb8.as_raw().len(), Urgb.nbytes() * (x * y) as usize);
        Self(Tensor::new(Urgb, &[y as usize, x as usize]).map(|_| rgb8.into_raw()))
    }
}

impl<T> Image<T> {
    #[inline]
    pub fn shape(&self) -> [usize; 2] {
        let &[h, w] = self.0.shape() else {
            unreachable!()
        };
        [w, h]
    }
}

impl<T> Image<T>
where
    T: Deref<Target = [u8]>,
{
    /// 图切片算法
    /// see <https://github.com/ggerganov/llama.cpp/blob/3952a221af54b8a6549bc2bd4a7363ef7ad3081e/examples/llava/clip.cpp#L1871>
    pub fn slice_uhd(
        &self,
        max_slices: usize,
        scale_resolution: usize,
        patch_size: usize,
    ) -> ImageGrid {
        let [w, h] = self.shape();

        // 图片按面积切分到图块，但不超过预设的上限
        let multiple = (w * h).div_ceil(scale_resolution.pow(2)).min(max_slices);
        let no_slice = multiple <= 1;

        ImageGrid {
            grid: if no_slice {
                None
            } else {
                let [grid_x, grid_y] = find_best_grid([w, h], max_slices, multiple);
                let [patch_w, patch_h] =
                    refine_patch_size([w, h], [grid_x, grid_y], scale_resolution, patch_size, true);

                let tiled = self
                    .bicubic_resize([grid_x * patch_w, grid_y * patch_h])
                    .0
                    .tile(1, &[grid_x, patch_w])
                    .tile(0, &[grid_y, patch_h])
                    .transpose(&[2, 1]);
                Some(tiled)
            },
            whole: self.bicubic_resize(find_best_resize(
                [w, h],
                scale_resolution,
                patch_size,
                no_slice,
            )),
        }
    }

    /// 双三次插值缩放
    fn bicubic_resize(&self, [w_, h_]: [usize; 2]) -> Image<Blob> {
        assert_eq!(self.0.dt(), Urgb);

        let mut ans = Image(Tensor::new(Urgb, &[h_, w_]).map(Blob::new));
        let data = ans.0.get_mut();
        let ptr = data.as_mut_ptr() as usize;
        let len = data.len();

        let [w, h] = self.shape();
        let data = &**self.0.get();
        let kw = w as fdim / w_ as fdim;
        let kh = h as fdim / h_ as fdim;

        (0..h_ * w_).into_par_iter().for_each(|i| {
            let dst = unsafe { from_raw_parts_mut(ptr as *mut u8, len) };
            let y_ = i / w_;
            let x_ = i % w_;

            let x = kw * x_ as fdim;
            let y = kh * y_ as fdim;
            let dx = x - x.floor();
            let dy = y - y.floor();
            // 在原图中的坐标
            let x = x as isize;
            let y = y as isize;
            // 插值循环
            for rgb in 0..3 {
                let mut c = [0.; 4];
                for ic in 0..=3 {
                    let data = |y: isize, x: isize| {
                        let x = x.clamp(0, w as isize - 1) as usize;
                        let y = y.clamp(0, h as isize - 1) as usize;
                        data[(y * w + x) * 3 + rgb] as fdim
                    };
                    c[ic as usize] = calc(|i| data(y - 1 + ic, x - 1 + i as isize), dx);

                    let cc = calc(|i| c[i], dy);
                    dst[(y_ * w_ + x_) * 3 + rgb] = cc.round().clamp(0., 255.) as _;
                }
            }
        });

        fn calc(c: impl Fn(usize) -> fdim, d: fdim) -> fdim {
            let a0 = c(1);
            let d0 = c(0) - a0;
            let d2 = c(2) - a0;
            let d3 = c(3) - a0;

            let mut ans = a0;
            let mut mul = d;

            ans += (-d0 / 3. + d2 - d3 / 6.) * mul;
            mul *= d;

            ans += (d0 / 2. + d2 / 2.) * mul;
            mul *= d;

            ans + (-d0 / 6. - d2 / 2. + d3 / 6.) * mul
        }

        ans
    }

    /// NHWC rgb Tensor -> NCHW value Tensor
    #[inline]
    pub fn to_nchw(&self) -> Tensor<Blob> {
        let src = self
            .0
            .destruct_array()
            .map(|t| &**t)
            .transpose(&[2, 0, 1])
            .tile(0, &[1, 3]);

        let mut ans = Tensor::new(src.dt(), src.shape()).map(Blob::new);
        rearrange(&mut ans, &src);
        ans
    }
}

impl ImageGrid {
    #[inline]
    pub fn whole(&self) -> &Image<Blob> {
        &self.whole
    }

    pub fn grid(&self) -> [usize; 2] {
        if let Some(grid) = &self.grid {
            let &[y, x, _, _] = grid.shape() else {
                unreachable!()
            };
            [x, y]
        } else {
            [0, 0]
        }
    }

    pub fn patch(&self, x: usize, y: usize) -> Image<&[u8]> {
        Image(
            self.grid
                .as_ref()
                .unwrap()
                .map_slice()
                .index(0, y)
                .index(0, x),
        )
    }

    /// [urgb] 转 [frgb]
    pub fn normalize(&self, mean: frgb, std: frgb) -> Self {
        Self {
            grid: self.grid.as_ref().map(|data| normalize(data, mean, std)),
            whole: Image(normalize(&self.whole.0, mean, std)),
        }
    }
}

/// 舍入除法
fn div_round(length: usize, patch_size: usize) -> usize {
    ((length as fdim / patch_size as fdim).round() as usize).max(1)
}

/// 缩放图片尺寸，使图片可按像素分到图块
fn find_best_resize(
    [mut w, mut h]: [usize; 2],
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> [usize; 2] {
    // 如果允许放大或需要缩小
    if allow_upscale || w * h > scale_resolution.pow(2) {
        // 保持宽高比缩放尺寸，使面积等于 `sr²`
        let r = (w as fdim / h as fdim).sqrt();
        w = (scale_resolution as fdim * r) as _;
        h = (scale_resolution as fdim / r) as _;
    }
    // 确保宽高整除图块尺寸
    [w, h].map(|l| div_round(l, patch_size) * patch_size)
}

/// 找到划分图块的最佳方式
fn find_best_grid(size: [usize; 2], max_slices: usize, multiple: usize) -> [usize; 2] {
    /// 宽高比的对数
    #[inline(always)]
    fn ln_ratio([w, h]: [usize; 2]) -> fdim {
        (w as fdim / h as fdim).ln()
    }

    let target = ln_ratio(size);

    // 候选分块数量可以是预设分块数量的相邻整数
    (multiple - 1..=multiple + 1)
        // 避免不分块或超过分块数量上限
        .filter(|m| (2..=max_slices).contains(m))
        // 对分块数量分解因数
        .flat_map(|num_grids| {
            (1..=num_grids)
                .filter(move |&w| num_grids % w == 0)
                .map(move |w| [w, num_grids / w])
        })
        // 找到与目标宽高比最接近的方案
        // TODO 确认此结论：正的浮点数编码排序与浮点数排序一致
        .min_by_key(|&grid| (target - ln_ratio(grid)).abs().to_bits())
        .unwrap()
}

/// 为每个图块优化尺寸
fn refine_patch_size(
    [w, h]: [usize; 2],
    [gx, gy]: [usize; 2],
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> [usize; 2] {
    find_best_resize(
        [div_round(w, gx), div_round(h, gy)],
        scale_resolution,
        patch_size,
        allow_upscale,
    )
}

/// 将整型表示的 rgb 值转换为归一化浮点表示
fn normalize<T>(data: &Tensor<T>, mean: frgb, std: frgb) -> Tensor<Blob>
where
    T: Deref<Target = [u8]>,
{
    assert_eq!(data.dt(), Urgb);
    let mut ans = data.cast(Frgb).map(Blob::new);

    let ([], src, []) = (unsafe { data.get().align_to::<urgb>() }) else {
        unreachable!()
    };
    let ([], dst, []) = (unsafe { ans.get_mut().align_to_mut::<frgb>() }) else {
        unreachable!()
    };

    for (dst, src) in zip(dst, src) {
        for (dst, &src, mean, std) in izip!(dst, src, mean, std) {
            *dst = (src as f32 / 255. - mean) / std;
        }
    }

    ans
}

#[test]
fn test() {
    use std::time::Instant;

    let Some(picture) = test_utils::image() else {
        return;
    };

    let time = Instant::now();
    let image = Image::load(picture);
    println!("load image {:?}", time.elapsed());

    let time = Instant::now();
    let slices = image
        .slice_uhd(9, 448, 14)
        .normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    println!("slice image {:?}", time.elapsed());

    let [x, y] = slices.grid();
    let whole = slices.whole();
    let [w, h] = whole.shape();
    let t = whole.to_nchw();
    println!("whole: {w}x{h}: {:?}/{:?}", t.shape(), t.strides());
    for i in 0..y {
        for j in 0..x {
            let patch = slices.patch(j, i);
            let [w, h] = patch.shape();
            let t = patch.to_nchw();
            println!("patch[{i}, {j}] {w}x{h}: {:?}/{:?}", t.shape(), t.strides());
        }
    }
}
