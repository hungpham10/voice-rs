//! Nạp WAV + tiền xử lý cho ContentVec, dùng chung cho `embed` và `embed_batch`.
//!
//! Sao chép đúng pipeline của Applio `create_reference.py`:
//! 1. chuẩn hoá đỉnh (với PCM int),
//! 2. cắt về bội số của 320 (hop length),
//! 3. reflect-pad 40 mẫu mỗi bên (padding conv của HuBERT).
//!
//! Ở đây **không** resample — dữ liệu phải đã là 16 kHz mono (khâu Python lo việc
//! đó, xem `scripts/data/`).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Đọc WAV thành mono `f32` (chưa tiền xử lý). Trả `Vec<f32>`.
pub fn load_wav_mono(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;

    let mut header = [0u8; 44];
    file.read_exact(&mut header)?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }

    let format_tag = u16::from_le_bytes([header[20], header[21]]);
    let channels = u16::from_le_bytes([header[22], header[23]]);
    let sample_rate = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
    let bits_per_sample = u16::from_le_bytes([header[34], header[35]]);
    println!("  WAV spec: {sample_rate} Hz, {channels} ch, {bits_per_sample} bits, format tag {format_tag}");

    let mut pos = 12u64;
    let mut data_pos = None;
    let mut data_size = 0u32;
    let file_size = file.metadata()?.len();
    while pos + 8 <= file_size {
        file.seek(SeekFrom::Start(pos))?;
        let mut chunk_header = [0u8; 8];
        file.read_exact(&mut chunk_header)?;
        let chunk_id = &chunk_header[0..4];
        let chunk_size =
            u32::from_le_bytes([chunk_header[4], chunk_header[5], chunk_header[6], chunk_header[7]]);
        if chunk_id == b"data" {
            data_pos = Some(pos + 8);
            data_size = chunk_size;
            break;
        }
        pos += 8 + chunk_size as u64;
    }
    let data_pos = data_pos.ok_or("No data chunk found")?;

    file.seek(SeekFrom::Start(data_pos))?;
    let samples: Vec<f32> = match (format_tag, bits_per_sample) {
        (1, 16) => {
            let mut buf = vec![0u8; data_size as usize];
            file.read_exact(&mut buf)?;
            buf.chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / i16::MAX as f32)
                .collect()
        }
        (1, 32) => {
            let mut buf = vec![0u8; data_size as usize];
            file.read_exact(&mut buf)?;
            buf.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / i32::MAX as f32)
                .collect()
        }
        (3, 32) => {
            let mut buf = vec![0u8; data_size as usize];
            file.read_exact(&mut buf)?;
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        _ => return Err(format!("Unsupported format: tag={format_tag}, bits={bits_per_sample}").into()),
    };

    let mono = if channels == 2 {
        samples.chunks_exact(2).map(|ch| (ch[0] + ch[1]) * 0.5).collect()
    } else {
        samples
    };

    if sample_rate != 16000 {
        println!("  ⚠️  Sample rate is {sample_rate} Hz, expected 16000 Hz. Consider resampling.");
    }

    Ok(mono)
}

/// Tiền xử lý khớp Applio `create_reference.py` (xem doc đầu file).
pub fn preprocess_applio(mut samples: Vec<f32>) -> Vec<f32> {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1.0 {
        let scale = 1.0 / peak;
        samples.iter_mut().for_each(|s| *s *= scale);
    }

    let trimmed_len = (samples.len() / 320) * 320;
    samples.truncate(trimmed_len);

    let pad = 40;
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    for i in (0..pad).rev() {
        padded.push(samples[i % samples.len()]);
    }
    padded.extend_from_slice(&samples);
    for i in 0..pad {
        padded.push(samples[samples.len() - 1 - (i % samples.len())]);
    }
    padded
}

/// Ghi `.npy` `float32` C-order với shape 3 chiều (`.npy` v1.0).
pub fn save_npy3(
    path: &Path,
    values: &[f32],
    dims: [usize; 3],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        dims[0], dims[1], dims[2]
    );
    let prefix = 6 + 2 + 2;
    let padding = (64 - ((prefix + header.len() + 1) % 64)) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');

    let mut out = Vec::with_capacity(prefix + header.len() + values.len() * 4);
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}
