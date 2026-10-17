//! Đọc `.npz` (zip chứa nhiều `.npy`) — chỉ cần `.npy` **không nén** (stored).
//!
//! Dùng chung cho các ví dụ (`verify_dec`, `verify_backward`, `verify_lora`) qua
//! `mod common;`.

use std::collections::BTreeMap;
use std::path::Path;

pub type Npz = BTreeMap<String, (Vec<f32>, Vec<usize>)>;

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

/// Đọc WAV 16-bit PCM hoặc float32, trộn kênh về mono.
///
/// ⚠️ WAV test là **float32** (`wFormatTag = 3`). `bits` nằm ở `fmt[7]`, block
/// align ở `fmt[6]` — đọc sai là sinh toàn `-10`.
#[allow(dead_code)]
pub fn load_wav(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    let mut f = File::open(path)?;
    let mut h = [0u8; 44];
    f.read_exact(&mut h)?;
    let fmt = u16::from_le_bytes([h[20], h[21]]);
    let ch = u16::from_le_bytes([h[22], h[23]]);
    let bits = u16::from_le_bytes([h[34], h[35]]);
    let mut pos = 12u64;
    let (mut dp, mut ds) = (0u64, 0u64);
    let total = f.metadata()?.len();
    while pos + 8 <= total {
        f.seek(SeekFrom::Start(pos))?;
        let mut ch8 = [0u8; 8];
        f.read_exact(&mut ch8)?;
        let sz = u32::from_le_bytes([ch8[4], ch8[5], ch8[6], ch8[7]]) as u64;
        if &ch8[0..4] == b"data" {
            dp = pos + 8;
            ds = sz;
            break;
        }
        pos += 8 + sz;
    }
    f.seek(SeekFrom::Start(dp))?;
    let mut buf = vec![0u8; ds as usize];
    f.read_exact(&mut buf)?;
    let s: Vec<f32> = match (fmt, bits) {
        (1, 16) => buf
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        (3, 32) => buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => return Err(format!("format {fmt}/{bits} khong ho tro").into()),
    };
    Ok(if ch == 2 {
        s.chunks_exact(2).map(|c| (c[0] + c[1]) * 0.5).collect()
    } else {
        s
    })
}