//! Đọc `.npz` (zip chứa nhiều `.npy`) — chỉ cần `.npy` **không nén** (stored).
//!
//! Dùng để đối chiếu đầu ra `dec` với ground truth PyTorch
//! (subcommand `voice verify`).

use std::collections::BTreeMap;
use std::path::Path;

pub type Npz = BTreeMap<String, (Vec<f32>, Vec<usize>)>;

/// Đọc một file `.npz` (stored, không nén) về map tên → (giá trị f32, dims).
pub fn read_npz(path: &Path) -> Result<Npz, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let mut out = BTreeMap::new();

    // Duyệt local file header (PK\x03\x04), đọc data ngay sau header.
    let mut i = 0usize;
    while i + 30 <= bytes.len() {
        if &bytes[i..i + 4] != b"PK\x03\x04" {
            break;
        }
        let method = u16::from_le_bytes([bytes[i + 8], bytes[i + 9]]);
        let name_len = u16::from_le_bytes([bytes[i + 26], bytes[i + 27]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[i + 28], bytes[i + 29]]) as usize;
        let name = String::from_utf8_lossy(&bytes[i + 30..i + 30 + name_len]).to_string();
        let data_start = i + 30 + name_len + extra_len;

        if method != 0 {
            return Err("npz phải lưu stored (không nén)".into());
        }
        let (vals, dims, consumed) = parse_npy(&bytes[data_start..])?;
        out.insert(name.trim_end_matches(".npy").to_string(), (vals, dims));
        i = data_start + consumed;

        // bỏ qua phần đệm / next header
        while i + 4 <= bytes.len() && &bytes[i..i + 4] != b"PK\x03\x04" {
            i += 1;
        }
    }
    Ok(out)
}

fn parse_npy(buf: &[u8]) -> Result<(Vec<f32>, Vec<usize>, usize), Box<dyn std::error::Error>> {
    if &buf[0..6] != b"\x93NUMPY" {
        return Err("khong phai npy".into());
    }
    let major = buf[6];
    let (hoff, hlen) = if major == 1 {
        (10usize, u16::from_le_bytes([buf[8], buf[9]]) as usize)
    } else {
        (12usize, u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize)
    };
    let h = std::str::from_utf8(&buf[hoff..hoff + hlen])?;
    let shape = h
        .split("'shape'")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').nth(0))
        .unwrap_or("");
    let dims: Vec<usize> = shape
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let body = &buf[hoff + hlen..];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = body
        .chunks_exact(4)
        .take(n)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let consumed = hoff + hlen + n * 4;
    let _ = dims;
    Ok((vals, dims, consumed))
}
