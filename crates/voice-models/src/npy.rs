//! Đọc file `.npy` (float32, ≤2 chiều) — dùng để nạp trọng số VITS.
//!
//! Trọng số `dec` được export MỘT LẦN từ checkpoint VITS (đóng băng trên HF,
//! phát trong release bundle); mỗi tensor nằm ở một file `.npy` với đường dẫn
//! phẳng (`ups.0.weight.npy`), nên ở đây chỉ cần bóc header rồi đọc thẳng
//! `f32` little-endian.

use std::fs;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum NpyError {
    #[error("không phải file .npy")]
    NotNpy,
    #[error("header .npy hỏng: {0}")]
    BadHeader(String),
    #[error("dtype không hỗ trợ: {0} (chỉ '<f4')")]
    UnsupportedDtype(String),
    #[error("số chiều không hợp lệ: {0} (chỉ 1..=3)")]
    BadDims(usize),
}

/// Đọc mảng `.npy`, trả về `(shape, dữ liệu)`.
pub fn read_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), NpyError> {
    let bytes = fs::read(path).map_err(|e| NpyError::BadHeader(format!("{}: {e}", path.display())))?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(NpyError::NotNpy);
    }
    let major = bytes[6];
    let (header_off, header_len) = if major == 1 {
        (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize)
    } else {
        (
            12usize,
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        )
    };
    let header = std::str::from_utf8(&bytes[header_off..header_off + header_len])
        .map_err(|e| NpyError::BadHeader(e.to_string()))?;

    let descr = header
        .split("'descr'")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .ok_or_else(|| NpyError::BadHeader("thiếu 'descr'".into()))?;
    if descr != "<f4" {
        return Err(NpyError::UnsupportedDtype(descr.to_string()));
    }

    let dims_str = header
        .split("'shape'")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').nth(0))
        .ok_or_else(|| NpyError::BadHeader("thiếu 'shape'".into()))?;
    let dims: Vec<usize> = dims_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if dims.is_empty() || dims.len() > 3 {
        return Err(NpyError::BadDims(dims.len()));
    }

    let body = &bytes[header_off + header_len..];
    let mut data: Vec<f32> = body
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // np.save (Python) mặc định fortran_order=False cho mảng 1D/2D lưu từ
    // tensor PyTorch (contiguous theo hàng). Với vector 1D, thứ tự là tuần tự.
    let fortran = header.contains("'fortran_order': True");
    if fortran && dims.len() == 2 {
        let (r, c) = (dims[0], dims[1]);
        let mut out = vec![0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                out[i * c + j] = data[j * r + i];
            }
        }
        data = out;
    }

    data.truncate(dims.iter().product());
    Ok((dims, data))
}

/// Nạp tensor 3D từ `.npy`, kiểm tra shape khớp trước khi trả về.
pub fn tensor3(path: &Path, expect: [usize; 3]) -> Result<Vec<f32>, NpyError> {
    let (dims, data) = read_f32(path)?;
    if dims != expect {
        return Err(NpyError::BadHeader(format!(
            "{}: shape {:?} != {:?}",
            path.display(),
            dims,
            expect
        )));
    }
    Ok(data)
}

/// Nạp vector 1D từ `.npy`, kiểm tra shape khớp trước khi trả về.
pub fn tensor1(path: &Path, expect: usize) -> Result<Vec<f32>, NpyError> {
    let (dims, data) = read_f32(path)?;
    if dims != [expect] {
        return Err(NpyError::BadHeader(format!(
            "{}: shape {:?} != [{}]",
            path.display(),
            dims,
            expect
        )));
    }
    Ok(data)
}