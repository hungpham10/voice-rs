//! `f0_batch` — sinh f0 (RMVPE) cho cả manifest CSV, ghi `.f0.npy` co-location
//! cạnh features.
//!
//! ```bash
//! cargo run --release -p voice-f0 --bin f0_batch -- \
//!   --manifest data/vivos/train.csv --skip-existing
//! ```
//!
//! Model RMVPE nạp **một lần** rồi lặp qua manifest. Mỗi dòng:
//!   feats: `features/x.npy`  →  f0: `features/x.f0.npy`  (`[1, T]` @100 Hz, f32)
//!
//! Hợp đồng CSV giữa khâu dữ liệu (Python) và train (Rust) **không đổi**; f0 là
//! file side-car, train đọc tự động khi có.

use std::path::{Path, PathBuf};

use clap::Parser;
use voice_f0::Rmvpe;
use voice_mel::MelConfig;

#[derive(Parser)]
#[command(name = "f0_batch", about = "Sinh f0 RMVPE (@100 Hz) cho manifest CSV")]
struct Args {
    /// Manifest CSV (cột: split,speaker_id,utt_id,wav_path,feats_path,duration).
    #[arg(long)]
    manifest: PathBuf,
    /// Chỉ xử lý dòng có `split` này (train/val). Bỏ trống = tất cả.
    #[arg(long)]
    split: Option<String>,
    /// Bỏ qua câu đã có file features f0.
    #[arg(long, default_value_t = false)]
    skip_existing: bool,
    /// Chỉ in kế hoạch, không chạy model.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
    /// `cpu` | `cuda`.
    #[arg(long, default_value = "cpu")]
    device: String,
}

/// Một dòng manifest đã tách.
struct Row {
    split: String,
    utt_id: String,
    wav_path: String,
    feats_path: String,
}

/// Đọc manifest: bỏ header, tách 6 cột, cho phép dòng trống.
fn read_manifest(path: &Path) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || i == 0 {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        if cols.len() != 6 {
            return Err(format!("dòng {}: cần 6 cột, nhận {} — `{line}`", i + 1, cols.len()).into());
        }
        rows.push(Row {
            split: cols[0].to_string(),
            utt_id: cols[2].to_string(),
            wav_path: cols[3].to_string(),
            feats_path: cols[4].to_string(),
        });
    }
    Ok(rows)
}

/// Đọc wav mono (hỗ trợ 16/32-bit int + float32).
fn load_wav_mono(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| Ok(s? as f32 / i16::MAX as f32))
            .collect::<Result<_, hound::Error>>()?,
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| Ok(s? as f32 / i32::MAX as f32))
            .collect::<Result<_, hound::Error>>()?,
        (hound::SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<_, hound::Error>>()?,
        (fmt, bits) => return Err(format!("wav không hỗ trợ: {fmt:?} {bits}-bit").into()),
    };
    Ok(samples)
}

/// Ghi `.npy` v1, shape `[1, T]`, f32, little-endian.
fn write_npy_f32(path: &Path, t: usize, flat: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut hdr = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': (1, {t}), }}");
    while hdr.len() < 62 {
        hdr.push(' ');
    }
    hdr.push('\n');
    let mut f = std::fs::File::create(path)?;
    f.write_all(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0])?;
    f.write_all(&(hdr.len() as u16).to_le_bytes())?;
    f.write_all(hdr.as_bytes())?;
    f.write_all(
        &flat
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let rows = read_manifest(&args.manifest)?;

    let device = device_select(&args.device)?;
    let todo: Vec<&Row> = rows
        .iter()
        .filter(|r| args.split.as_deref().map(|s| r.split.eq_ignore_ascii_case(s)).unwrap_or(true))
        .filter(|r| {
            let f0_path = Path::new(&r.feats_path).with_extension("f0.npy");
            !(args.skip_existing && f0_path.exists())
        })
        .collect();

    println!(
        "== f0_batch: manifest {} · {} dòng · {} cần xử lý ==",
        args.manifest.display(),
        rows.len(),
        todo.len()
    );
    if args.dry_run {
        for r in &todo {
            println!("  {} → {}", r.wav_path, r.feats_path);
        }
        return Ok(());
    }
    if todo.is_empty() {
        println!("không có gì để làm");
        return Ok(());
    }

    let rmvpe = Rmvpe::new_with_device(device)?;
    let mel_cfg = MelConfig::rmvpe();
    println!("device: {:?}", rmvpe.device());

    for (i, r) in todo.iter().enumerate() {
        if !Path::new(&r.wav_path).exists() {
            return Err(format!("{} chưa có (wav)", r.wav_path).into());
        }
        let f0_path = Path::new(&r.feats_path).with_extension("f0.npy");
        if f0_path.exists() {
            continue;
        }
        println!("[{}/{}] {}", i + 1, todo.len(), r.utt_id);
        let samples = load_wav_mono(Path::new(&r.wav_path))?;
        let mel = mel_cfg.spectrogram(&samples)?;
        let f0 = rmvpe.forward_f0(&mel)?; // [T_f0] @100 Hz
        write_npy_f32(&f0_path, f0.len(), &f0)?;
        println!("     ghi {} ({} frame)", f0_path.display(), f0.len());
    }
    println!("✔ xong f0_batch");
    Ok(())
}

/// Chọn device cho RMVPE (CPU luôn có; CUDA chưa bật trong crate voice-f0).
fn device_select(name: &str) -> Result<burn::tensor::Device, Box<dyn std::error::Error>> {
    match name {
        "cpu" => Ok(burn::tensor::Device::flex()),
        other => Err(format!(
            "device '{other}' chưa hỗ trợ trong voice-f0 (chỉ cpu); \
             dùng voice-train --device cuda nếu cần GPU cho train"
        )
        .into()),
    }
}
