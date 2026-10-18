//! Đọc `.npz` (zip chứa nhiều `.npy`) — chỉ cần `.npy` **không nén** (stored).
//!
//! Re-export từ [`voice_vits::npz`] để duy nhất một nguồn đọc npz cho toàn bộ
//! các ví dụ verify.

pub use voice_vits::npz::{self, read_npz, Npz};

use std::path::Path;

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
