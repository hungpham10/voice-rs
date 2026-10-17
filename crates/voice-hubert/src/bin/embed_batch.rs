//! `embed_batch` — sinh features ContentVec cho cả manifest (§13.6.4c).
//!
//! Đọc manifest CSV hợp đồng (§13.6.4d) và ghi features `.npy` vào `feats_path`
//! của từng dòng. Nạp model **một lần** rồi lặp — tránh khởi động lại cho mỗi câu.
//!
//! ```bash
//! cargo run --release -p voice-hubert --bin embed_batch -- \
//!   --manifest data/vivos/train.csv --skip-existing
//! ```
//!
//! Lưu ý: đây là công cụ **dev**, không thuộc runtime suy luận. Model ONNX phải
//! có sẵn ở `sources/contentvec/` (do `scripts/fetch_assets.sh` nạp).

use std::path::{Path, PathBuf};

use clap::Parser;
use voice_hubert::wavio::{load_wav_mono, preprocess_applio, save_npy3};
use voice_hubert::HubertEmbedder;

#[derive(Parser)]
#[command(name = "embed_batch", about = "Sinh features ContentVec cho một manifest CSV")]
struct Args {
    /// Manifest CSV (cột: split,speaker_id,utt_id,wav_path,feats_path,duration).
    #[arg(long)]
    manifest: PathBuf,
    /// Chỉ xử lý dòng có `split` này (train/val). Bỏ trống = tất cả.
    #[arg(long)]
    split: Option<String>,
    /// Bỏ qua câu đã có file features.
    #[arg(long, default_value_t = false)]
    skip_existing: bool,
    /// Chỉ in kế hoạch, không chạy model.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
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
        if line.is_empty() {
            continue;
        }
        if i == 0 {
            continue; // header
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let rows = read_manifest(&args.manifest)?;

    let todo: Vec<&Row> = rows
        .iter()
        .filter(|r| args.split.as_deref().map(|s| r.split.eq_ignore_ascii_case(s)).unwrap_or(true))
        .filter(|r| !(args.skip_existing && Path::new(&r.feats_path).exists()))
        .collect();

    println!(
        "manifest {} · {} dòng · {} cần xử lý",
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

    let embedder = HubertEmbedder::new()?;
    println!("device: {:?}", embedder.device());

    for (i, r) in todo.iter().enumerate() {
        if !Path::new(&r.wav_path).exists() {
            return Err(format!("{} chưa có (wav)", r.wav_path).into());
        }
        if Path::new(&r.feats_path).exists() {
            continue;
        }
        println!("[{}/{}] {}", i + 1, todo.len(), r.utt_id);
        let samples = preprocess_applio(load_wav_mono(Path::new(&r.wav_path))?);
        let embeddings = embedder.forward(&samples)?;
        let dims: [usize; 3] = embeddings.shape().dims();
        let values = embeddings.into_data().as_slice::<f32>().unwrap().to_vec();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("{}: features chứa NaN/Inf", r.utt_id).into());
        }
        save_npy3(Path::new(&r.feats_path), &values, dims)?;
        println!("     ghi {} (shape {:?})", r.feats_path, dims);
    }
    println!("✔ xong");
    Ok(())
}
