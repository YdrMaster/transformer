use super::Tensor;
use ggus::ggml_quants::{bf16, digit_layout::types as primitive, f16};
use std::{fmt, ops::Deref};

pub trait DataFmt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result;
}

impl DataFmt for f16 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *self == f16::ZERO {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.to_f32())
        }
    }
}

impl DataFmt for bf16 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *self == bf16::ZERO {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.to_f32())
        }
    }
}

impl DataFmt for f32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *self == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{self:>9.3e}")
        }
    }
}

impl DataFmt for f64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *self == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{self:>9.3e}")
        }
    }
}

impl DataFmt for u32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *self == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{self:>6}")
        }
    }
}

impl DataFmt for u64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *self == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{self:>6}")
        }
    }
}

impl<Physical: Deref<Target = [u8]>> fmt::Display for Tensor<Physical> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.dt {
            primitive::F16 => self.map_slice().write_tensor::<f16>(&mut vec![], f),
            primitive::BF16 => self.map_slice().write_tensor::<bf16>(&mut vec![], f),
            primitive::F32 => self.map_slice().write_tensor::<f32>(&mut vec![], f),
            primitive::F64 => self.map_slice().write_tensor::<f64>(&mut vec![], f),
            primitive::U32 => self.map_slice().write_tensor::<u32>(&mut vec![], f),
            primitive::U64 => self.map_slice().write_tensor::<u64>(&mut vec![], f),
            _ => todo!(),
        }
    }
}

impl Tensor<&[u8]> {
    fn write_tensor<T: DataFmt>(
        &self,
        indices: &mut Vec<[usize; 2]>,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        match *self.shape() {
            [] => {
                write!(f, "<>")?;
                write_indices(f, indices)?;
                write_matrix(f, self.base().cast::<T>(), [1, 1], [1, 1])
            }
            [len] => {
                let &[stride] = self.strides() else {
                    unreachable!()
                };
                write!(f, "<{len}>")?;
                write_indices(f, indices)?;
                write_matrix(f, self.base().cast::<T>(), [len, 1], [stride, 1])
            }
            [rows, cols] => {
                let &[rs, cs] = self.strides() else {
                    unreachable!()
                };
                write!(f, "<{rows}x{cols}>")?;
                write_indices(f, indices)?;
                write_matrix(f, self.base().cast::<T>(), [rows, cols], [rs, cs])
            }
            [batch, ..] => {
                for i in 0..batch {
                    indices.push([i, batch]);
                    self.map_slice().index(0, i).write_tensor::<T>(indices, f)?;
                    indices.pop();
                }
                Ok(())
            }
        }
    }
}

fn write_matrix<T: DataFmt>(
    f: &mut fmt::Formatter,
    ptr: *const T,
    shape: [usize; 2],
    strides: [isize; 2],
) -> fmt::Result {
    let [rows, cols] = shape;
    let [rs, cs] = strides;
    for r in 0..rows as isize {
        for c in 0..cols as isize {
            unsafe { &*ptr.byte_offset(r * rs + c * cs) }.fmt(f)?;
            write!(f, " ")?;
        }
        writeln!(f)?;
    }
    Ok(())
}

fn write_indices(f: &mut fmt::Formatter, indices: &[[usize; 2]]) -> fmt::Result {
    for &[i, b] in indices {
        write!(f, ", {i}/{b}")?;
    }
    writeln!(f)
}

#[test]
fn test_fmt() {
    use primitive::F32;
    use std::slice::from_raw_parts;

    let shape = [2, 3, 4];
    let data = Vec::from_iter((0..24).map(|x| x as f32));
    let data = unsafe { from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(&*data)) };
    // [2, 3, 4]
    let t = Tensor::new(F32, &shape).map(|_| data);
    println!("{t}");
    // [2, 3, 2, 2]
    let t = t.tile(2, &[2, 2]);
    println!("{t}");
    // [6, 4]
    let t = t.merge(0..2).unwrap().merge(1..3).unwrap();
    println!("{t}");
    // [4, 6]
    let t = t.transpose(&[1, 0]);
    println!("{t}");
    // [2, 3]
    let t = t.slice(0, 0, 2, 2).slice(1, 1, 2, 3);
    println!("{t}");
    // [3, 2]
    let t = t.transpose(&[1, 0]);
    println!("{t}");
    // [3, 1, 2]
    let t = t.tile(0, &[3, 1]);
    println!("{t}");
    // [3, 3, 2]
    let t = t.slice(1, 0, 0, 3);
    println!("{t}");
}
