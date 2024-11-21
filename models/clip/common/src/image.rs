use gguf::ggml_quants::digit_layout::layout;
use image::ImageReader;
use itertools::izip;
use std::{ops::Deref, path::Path};
use tensor::{rearrange, Blob, Tensor};

#[repr(transparent)]
pub struct Image<T>(Tensor<T>);

pub struct ImageGrid {
    grid: Option<Tensor<Blob>>,
    whole: Image<Blob>,
}

layout!(Urgb u(8)     ; 3);
layout!(Frgb e(8)m(23); 3);

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
                let mut ans = Tensor::new(Urgb, tiled.shape()).map(Blob::new);
                rearrange(&mut ans, &tiled.map_slice());

                Some(ans)
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

        let mut ans_ = Image(Tensor::new(Urgb, &[h_, w_]).map(Blob::new));
        let ans = ans_.0.get_mut();

        let [w, h] = self.shape();
        let data = self.0.get();
        let kw = w as f64 / w_ as f64;
        let kh = h as f64 / h_ as f64;

        for y_ in 0..h_ {
            for x_ in 0..w_ {
                let x = kw * x_ as f64;
                let y = kh * y_ as f64;
                let dx = x - x.floor();
                let dy = y - y.floor();
                // 在原图中的坐标
                let x = x as isize;
                let y = y as isize;
                // 插值循环
                for rgb in 0..3 {
                    let mut c = [0.0f64; 4];
                    for ic in 0..=3 {
                        fn calc(c: impl Fn(usize) -> f64, d: f64) -> f64 {
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

                        let data = |y: isize, x: isize| {
                            let x = x.clamp(0, w as isize - 1) as usize;
                            let y = y.clamp(0, h as isize - 1) as usize;
                            data[(y * w + x) * 3 + rgb] as f64
                        };
                        c[ic as usize] = calc(|i| data(y - 1 + ic, x - 1 + i as isize), dx);

                        let cc = calc(|i| c[i], dy);
                        ans[(y_ * w_ + x_) * 3 + rgb] = cc.round().clamp(0., 255.) as _;
                    }
                }
            }
        }

        ans_
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

    /// [u8;3] 转 [f32;3]
    pub fn normalize(&self, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            grid: self.grid.as_ref().map(|data| normalize(data, mean, std)),
            whole: Image(normalize(&self.whole.0, mean, std)),
        }
    }
}

/// 舍入除法
fn div_round(length: usize, patch_size: usize) -> usize {
    ((length as f64 / patch_size as f64).round() as usize).max(1)
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
        let r = (w as f64 / h as f64).sqrt();
        w = (scale_resolution as f64 * r) as _;
        h = (scale_resolution as f64 / r) as _;
    }
    // 确保宽高整除图块尺寸
    [w, h].map(|l| div_round(l, patch_size) * patch_size)
}

/// 找到划分图块的最佳方式
fn find_best_grid(size: [usize; 2], max_slices: usize, multiple: usize) -> [usize; 2] {
    /// 宽高比的对数
    #[inline(always)]
    fn ln_ratio([w, h]: [usize; 2]) -> f64 {
        (w as f64 / h as f64).ln()
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
fn normalize<T>(data: &Tensor<T>, mean: [f32; 3], std: [f32; 3]) -> Tensor<Blob>
where
    T: Deref<Target = [u8]>,
{
    assert_eq!(data.dt(), Urgb);
    let mut ans = Tensor::new(Frgb, data.shape()).map(Blob::new);

    let src = &**data.get();
    let ([], dst, []) = (unsafe { ans.get_mut().align_to_mut::<f32>() }) else {
        unreachable!()
    };

    assert_eq!(dst.len(), src.len());
    assert_eq!(dst.len() % 3, 0);
    for i in 0..dst.len() / 3 {
        for (dst, &src, mean, std) in izip!(&mut dst[i * 3..], &src[i * 3..], mean, std) {
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
    let [w, h] = slices.whole().shape();
    println!("whole: {w}x{h}");
    for j in 0..y {
        for i in 0..x {
            let [w, h] = slices.patch(i, j).shape();
            println!("patch[{i}, {j}] {w}x{h}")
        }
    }
}
