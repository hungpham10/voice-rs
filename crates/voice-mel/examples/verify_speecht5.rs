//! Verify `MelConfig::speecht5()` against the HF ground truth in
//! `sources/ref_mel.npy` (ground truth log-mel đã đóng băng, sinh một lần từ
//! `transformers.SpeechT5FeatureExtractor._extract_mel_features`).
//!
//! Run:  cargo run --release -p voice-mel --example verify_speecht5

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use voice_mel::MelConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wav: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sources/422-122949-0031.wav".into())
        .into();
    let npy = Path::new("sources/ref_mel.npy");

    let samples = load_wav(&wav)?;
    println!("wav: {} samples ({} Hz)", samples.len(), 16000);

    let cfg = MelConfig::speecht5();
    let mel = cfg.spectrogram(&samples)?;
    let n_mels = mel.len();
    let n_frames = mel[0].len();
    println!("rust mel: [{} mels, {} frames]", n_mels, n_frames);

    let (ref_mels, ref_frames, ref_data) = load_npy_f32(npy)?;
    println!("ref  mel: [{} mels, {} frames]", ref_mels, ref_frames);

    if n_mels != ref_mels || n_frames != ref_frames {
        return Err(format!(
            "shape mismatch: rust [{}x{}] vs ref [{}x{}]",
            n_mels, n_frames, ref_mels, ref_frames
        )
        .into());
    }

    // ref_data is row-major [n_mels, n_frames]
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut worst = (0usize, 0usize);
    for m in 0..n_mels {
        for j in 0..n_frames {
            let a = mel[m][j] as f64;
            let b = ref_data[m * n_frames + j] as f64;
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
                worst = (m, j);
            }
            let denom = b.abs().max(1e-8);
            max_rel = max_rel.max(d / denom);
        }
    }

    println!("max abs diff = {max_abs:.3e}");
    println!("max rel diff = {max_rel:.3e}");
    println!("worst cell: m={} frame={}", worst.0, worst.1);

    let pass = max_abs < 1e-5;
    println!("{}", if pass { "PASS" } else { "FAIL" });
    std::process::exit(if pass { 0 } else { 1 });
}

/// Minimal `.npy` reader for little-endian float32 arrays up to rank 2.
fn load_npy_f32(path: &Path) -> Result<(usize, usize, Vec<f32>), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = bytes[6];
    let (header_off, header_len) = if major == 1 {
        (
            10usize,
            u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        )
    } else {
        (12usize, u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize)
    };
    let header = std::str::from_utf8(&bytes[header_off..header_off + header_len])?;

    let descr = header
        .split("'descr'")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .ok_or("missing 'descr'")?;
    if descr != "<f4" {
        return Err(format!("expected '<f4', got '{}'", descr).into());
    }

    let fortran = header.contains("'fortran_order': True");

    let shape = header
        .split("'shape'")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').nth(0))
        .ok_or("missing 'shape'")?;
    let dims: Vec<usize> = shape
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if dims.len() != 2 {
        return Err(format!("expected 2D, got {} dims", dims.len()).into());
    }

    let body = &bytes[header_off + header_len..];
    let mut data: Vec<f32> = body
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    data.truncate(dims[0] * dims[1]);

    if fortran {
        // column-major -> row-major
        let mut out = vec![0f32; dims[0] * dims[1]];
        for r in 0..dims[0] {
            for c in 0..dims[1] {
                out[r * dims[1] + c] = data[c * dims[0] + r];
            }
        }
        return Ok((dims[0], dims[1], out));
    }
    Ok((dims[0], dims[1], data))
}

fn load_wav(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;

    let mut header = [0u8; 44];
    file.read_exact(&mut header)?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }
    let format_tag = u16::from_le_bytes([header[20], header[21]]);
    let channels = u16::from_le_bytes([header[22], header[23]]);
    let bits_per_sample = u16::from_le_bytes([header[34], header[35]]);

    let mut pos = 12u64;
    let mut data_pos = None;
    let mut data_size = 0;
    let file_size = file.metadata()?.len();
    while pos + 8 <= file_size {
        file.seek(SeekFrom::Start(pos))?;
        let mut chunk_header = [0u8; 8];
        file.read_exact(&mut chunk_header)?;
        let chunk_id = &chunk_header[0..4];
        let chunk_size = u32::from_le_bytes([chunk_header[4], chunk_header[5], chunk_header[6], chunk_header[7]]);
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
        (3, 32) => {
            let mut buf = vec![0u8; data_size as usize];
            file.read_exact(&mut buf)?;
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        _ => {
            return Err(
                format!("Unsupported format: tag={}, bits={}", format_tag, bits_per_sample).into(),
            )
        }
    };

    let mono = if channels == 2 {
        samples.chunks_exact(2).map(|ch| (ch[0] + ch[1]) * 0.5).collect()
    } else {
        samples
    };
    Ok(mono)
}
