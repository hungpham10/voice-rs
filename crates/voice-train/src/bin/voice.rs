//! CLI hợp nhất **một lệnh** cho đường VC 1→1 (§13.6.4a).
//!
//! ```text
//! voice embed    --manifest train.csv [--skip-existing]   # hoặc --wav/--out cho 1 file
//! voice train    --train-manifest ... --val-manifest ... [--device cpu|cuda]
//! voice validate --val-manifest ... --adapter ... [--gate gate.toml]
//! voice generate --feats ... --adapter ... [--out-wav render.wav]
//! voice diag     --target ... --gen ... [--f0 f0.npy]
//! voice verify   --dec-dir sources/vits_ljs_dec [--device cpu|cuda]
//! ```
//!
//! - `embed` = sinh features ContentVec (HuBERT) — model nạp **một lần** rồi lặp
//!   qua manifest (trước đây là bin `embed`/`embed_batch` của `voice-hubert`).
//! - `train` = Luồng 1 (cầu nối nội dung nhiều giọng) — đường huấn luyện duy nhất;
//!   chỉ `Adapter` học, `dec` đóng băng.
//! - `validate` thoát với mã **1** khi vi phạm cổng — dùng trực tiếp trong CI/Colab.
//! - `generate` = **suy diễn thuần** feats + adapter → wav (mặc định cả câu).
//!
//! `train` hỗ trợ **checkpoint định kỳ** (`--checkpoint-every`) và **đẩy kết quả
//! lên HuggingFace Hub** (`--push-to-hub`, mặc định tắt).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use burn::tensor::{Tensor, TensorData};
use clap::{Args, Parser, Subcommand};
use voice_lib::vc::Adapter;
use voice_mel::MelConfig;
use voice_vits::dec::Generator;
use voice_vits::npz::read_npz;
use voice_train::dataset::Dataset;
use voice_train::hub::HubConfig;
use voice_train::train::TrainConfig;
use voice_train::{device, eval, manifest, mel, render, train};

#[derive(Parser)]
#[command(
    name = "voice",
    version,
    about = "Huấn luyện + suy diễn VC 1→1 — một lệnh, nhiều subcommand"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Sinh features ContentVec cho manifest CSV hoặc 1 wav
    Embed(EmbedArgs),
    /// Huấn luyện Luồng 1 (nhiều giọng, chỉ `Adapter` học, `dec` đóng băng)
    Train(TrainArgs),
    /// Đánh giá + cổng PASS/FAIL (mã thoát 1 khi FAIL)
    Validate(EvalArgs),
    /// Suy diễn: feats + adapter → wav (không huấn luyện; mặc định cả câu)
    Generate(GenerateArgs),
    /// Chẩn đoán rè/nhiễu: mel-L1 theo band + voiced/unvoiced (PR-B1)
    Diag(DiagArgs),
    /// Verify `dec` (Rust) vs PyTorch ground truth npz — bước kiểm tra trước khi train
    Verify(VerifyArgs),
}

/// Tham số `embed` — HuBERT/ContentVec → features `.npy`.
///
/// Dùng **manifest** (từng dòng 1 câu, model nạp 1 lần) hoặc **1 wav** đơn lẻ.
#[derive(Args)]
struct EmbedArgs {
    /// Manifest CSV (cột: split,speaker_id,utt_id,wav_path,feats_path,duration).
    #[arg(long, conflicts_with_all = ["wav", "out"])]
    manifest: Option<PathBuf>,
    /// 1 wav đơn lẻ 16 kHz mono (song song với `--manifest`).
    #[arg(long)]
    wav: Option<PathBuf>,
    /// File `.npy` ghi ra (khi `--wav`).
    #[arg(long, requires = "wav")]
    out: Option<PathBuf>,
    /// Chỉ xử lý dòng có `split` này (train/val). Bỏ trống = tất cả.
    #[arg(long)]
    split: Option<String>,
    /// Bỏ qua câu đã có file features.
    #[arg(long, default_value_t = false)]
    skip_existing: bool,
    /// Chỉ in kế hoạch, không chạy model.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
    /// `cpu` | `cuda` (mặc định `cpu` cho dev; máy có NVIDIA có thể dùng cuda).
    #[arg(long, default_value = "cpu")]
    device: String,
}

#[derive(Args)]
struct GenerateArgs {
    #[arg(long, default_value = "sources/vits_ljs_dec")]
    dec_dir: PathBuf,
    /// Features ContentVec (`.npy`) — thường là của cả câu.
    #[arg(long)]
    feats: PathBuf,
    /// Adapter đã huấn luyện (`.bpk`).
    #[arg(long)]
    adapter: PathBuf,
    #[arg(long, default_value = "render.wav")]
    out_wav: PathBuf,
    /// `0` (mặc định) = cả câu; `>0` = giới hạn số frame latent.
    #[arg(long, default_value_t = 0)]
    seg: usize,
    /// `cpu` | `cuda` (mặc định `cpu` cho dev; release binary ưu tiên CUDA).
    #[arg(long, default_value = "cpu")]
    device: String,
}

/// Tuỳ chọn đẩy kết quả lên HuggingFace Hub — **mặc định tắt**.
#[derive(Args, Debug)]
struct HubArgs {
    /// Bật đẩy kết quả lên HuggingFace Hub sau khi huấn luyện.
    #[arg(long, default_value_t = false)]
    push_to_hub: bool,
    /// Repo đích dạng `user/name` (bắt buộc khi bật `--push-to-hub`).
    #[arg(long)]
    hub_repo: Option<String>,
    /// Token Hub (mặc định lấy biến môi trường `HF_TOKEN`).
    #[arg(long)]
    hf_token: Option<String>,
    /// Tiền tố đường dẫn trong repo (vd `bridge/`); rỗng = để ở gốc.
    #[arg(long, default_value = "")]
    hub_prefix: String,
}

impl HubArgs {
    fn resolve(&self) -> Result<Option<HubConfig>, String> {
        HubConfig::resolve(
            self.push_to_hub,
            self.hub_repo.clone(),
            self.hf_token.clone(),
            self.hub_prefix.clone(),
        )
    }
}

/// Tham số `train` — Luồng 1: nhiều giọng từ manifest CSV.
#[derive(Args)]
struct TrainArgs {
    /// Manifest các câu train (CSV hợp đồng).
    #[arg(long)]
    train_manifest: PathBuf,
    /// Manifest các câu val.
    #[arg(long)]
    val_manifest: PathBuf,
    #[arg(long, default_value = "sources/vits_ljs_dec")]
    dec_dir: PathBuf,
    #[arg(long, default_value = "sources/features/bridge")]
    out_dir: PathBuf,
    /// Số frame latent mỗi mẫu (tự co theo câu ngắn nhất).
    #[arg(long, default_value_t = 86)]
    seg: usize,
    /// Số câu mỗi bước.
    #[arg(long, default_value_t = 8)]
    batch: usize,
    #[arg(long, default_value_t = 500)]
    steps: usize,
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    /// Cứ bao nhiêu bước đo val một lần.
    #[arg(long, default_value_t = 50)]
    val_every: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// `0` = tắt; `>0` = cứ mỗi N bước lưu `checkpoint.bpk` trong `--out-dir`.
    #[arg(long, default_value_t = 0)]
    checkpoint_every: usize,
    /// `cpu` | `cuda` (mặc định `cpu` cho dev; release binary ưu tiên CUDA).
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Bật NSF-lite: unfreeze `conv_src` + bơm excitation từ f0 (yêu cầu .f0.npy).
    #[arg(long, default_value_t = false)]
    nsf: bool,
    #[command(flatten)]
    hub: HubArgs,
}

#[derive(Args)]
struct EvalArgs {    /// Manifest các câu val.
    #[arg(long)]
    val_manifest: PathBuf,
    /// Adapter đã huấn luyện (`.bpk`).
    #[arg(long)]
    adapter: PathBuf,
    #[arg(long, default_value = "sources/vits_ljs_dec")]
    dec_dir: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
    /// File ngưỡng `key = value` (mặc định: ngưỡng trong code).
    #[arg(long)]
    gate: Option<PathBuf>,
    /// Ghi báo cáo từng câu ra CSV.
    #[arg(long)]
    out_report: Option<PathBuf>,
    /// Ghi bản tóm tắt (metrics + gate + pass) ra JSON — dùng làm golden cho lần sau.
    #[arg(long)]
    out_json: Option<PathBuf>,
    /// So với `summary.json` của version trước; thoái lui quá ngưỡng = FAIL.
    #[arg(long)]
    baseline: Option<PathBuf>,
    /// Ngưỡng thoái lui tương đối so với baseline (mặc định 5%).
    #[arg(long, default_value_t = 0.05)]
    max_regression: f64,
    /// Ghi audio sinh cho từng câu vào thư mục này.
    #[arg(long)]
    out_audio_dir: Option<PathBuf>,
    #[arg(long)]
    max_mel_l1: Option<f64>,
    #[arg(long)]
    max_speaker_gap: Option<f64>,
    #[arg(long)]
    min_rms: Option<f64>,
    #[arg(long)]
    max_rms: Option<f64>,
    #[arg(long)]
    max_drift: Option<f64>,
}

/// Tham số `diag` — chẩn đoán rè giữa wav sinh ra (gen) và wav đích (target).
#[derive(Parser)]
struct DiagArgs {
    /// Wav đích (ground truth, mọi sample rate — tự resample về 22.05k).
    #[arg(long)]
    target: PathBuf,
    /// Wav sinh ra (từ `render`/`train overfit`, chuẩn 22.05 kHz).
    #[arg(long)]
    gen: PathBuf,
    /// f0 RMVPE `.npy` @100 Hz (`[1,T]`) để tách voiced/unvoiced (tuỳ chọn).
    #[arg(long)]
    f0: Option<PathBuf>,
    /// Ghi CSV từng frame: frame,overall,low,mid,high[,voiced].
    #[arg(long)]
    out_csv: Option<PathBuf>,
}

/// Tham số `verify` — đối chiếu `dec` (Burn) với ground truth PyTorch npz.
#[derive(Args)]
struct VerifyArgs {
    /// Thư mục dec chứa config.json + dec/*.npy + ref_audio.npz + ref_input.npz.
    #[arg(long, default_value = "sources/vits_ljs_dec")]
    dec_dir: PathBuf,
    /// `cpu` | `cuda`.
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Embed(a) => run_embed(a),
        Cmd::Train(a) => run_train(a),
        Cmd::Validate(a) => run_validate(a),
        Cmd::Generate(a) => run_generate(a),
        Cmd::Diag(a) => run_diag(a),
        Cmd::Verify(a) => run_verify(a),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("❌ {e}");
            ExitCode::FAILURE
        }
    }
}

/// Một dòng manifest đã tách cho `embed`.
struct EmbedRow {
    split: String,
    utt_id: String,
    wav_path: String,
    feats_path: String,
}

/// Đọc manifest cho `embed`: bỏ header, tách 6 cột, cho phép dòng trống.
fn read_embed_manifest(path: &Path) -> Result<Vec<EmbedRow>, Box<dyn std::error::Error>> {
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
        rows.push(EmbedRow {
            split: cols[0].to_string(),
            utt_id: cols[2].to_string(),
            wav_path: cols[3].to_string(),
            feats_path: cols[4].to_string(),
        });
    }
    Ok(rows)
}

/// Sinh features cho 1 wav: HuBERT/ContentVec → `.npy` `[1, 768, T]`.
fn embed_one_wav(
    embedder: &voice_hubert::HubertEmbedder,
    wav: &Path,
    feats: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let samples = voice_hubert::wavio::preprocess_applio(voice_hubert::wavio::load_wav_mono(wav)?);
    let embeddings = embedder.forward(&samples)?;
    let dims: [usize; 3] = embeddings.shape().dims();
    let values = embeddings.into_data().as_slice::<f32>().unwrap().to_vec();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("{}: features chứa NaN/Inf", wav.display()).into());
    }
    if let Some(parent) = feats.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    voice_hubert::wavio::save_npy3(feats, &values, dims)?;
    println!("  ghi {} (shape {:?})", feats.display(), dims);
    Ok(())
}

/// Subcommand `embed` — hai chế độ: manifest CSV (lặp từng câu) hoặc 1 wav.
fn run_embed(a: EmbedArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    match (&a.manifest, &a.wav) {
        (Some(_), None) => {}
        (None, Some(w)) => {
            let out = a.out.as_ref().ok_or("--wav cần --out <feats.npy>")?;
            let device = device::select(&a.device)?;
            let embedder = voice_hubert::HubertEmbedder::new_with_device(device)?;
            println!("== voice embed (device {:?}) ==", embedder.device());
            embed_one_wav(&embedder, w, out)?;
            return Ok(ExitCode::SUCCESS);
        }
        _ => {
            return Err("cần --manifest <csv> HOẶC --wav <in.wav> --out <feats.npy>".into());
        }
    }

    let manifest = a.manifest.as_ref().unwrap();
    let rows = read_embed_manifest(manifest)?;
    let todo: Vec<&EmbedRow> = rows
        .iter()
        .filter(|r| {
            a.split
                .as_deref()
                .map(|s| r.split.eq_ignore_ascii_case(s))
                .unwrap_or(true)
        })
        .filter(|r| !(a.skip_existing && Path::new(&r.feats_path).exists()))
        .collect();

    println!(
        "== voice embed: manifest {} · {} dòng · {} cần xử lý ==",
        manifest.display(),
        rows.len(),
        todo.len()
    );
    if a.dry_run {
        for r in &todo {
            println!("  {} → {}", r.wav_path, r.feats_path);
        }
        return Ok(ExitCode::SUCCESS);
    }
    if todo.is_empty() {
        println!("không có gì để làm");
        return Ok(ExitCode::SUCCESS);
    }

    let device = device::select(&a.device)?;
    let embedder = voice_hubert::HubertEmbedder::new_with_device(device)?;
    println!("device: {:?}", embedder.device());

    for (i, r) in todo.iter().enumerate() {
        if !Path::new(&r.wav_path).exists() {
            return Err(format!("{} chưa có (wav)", r.wav_path).into());
        }
        println!("[{}/{}] {}", i + 1, todo.len(), r.utt_id);
        embed_one_wav(&embedder, Path::new(&r.wav_path), Path::new(&r.feats_path))?;
    }
    println!("✔ xong embed");
    Ok(ExitCode::SUCCESS)
}

fn run_train(a: TrainArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let hub = a.hub.resolve()?;
    let device = device::select(&a.device)?;

    let train_entries = manifest::read(&a.train_manifest)?;
    let val_entries = manifest::read(&a.val_manifest)?;
    if train_entries.is_empty() {
        return Err(format!("{} rỗng", a.train_manifest.display()).into());
    }
    if val_entries.is_empty() {
        return Err(format!("{} rỗng", a.val_manifest.display()).into());
    }

    println!("== nạp dataset ==");
    let train_ds = Dataset::load(&train_entries)?;
    let val_ds = Dataset::load(&val_entries)?;

    let cfg = TrainConfig {
        dec_dir: a.dec_dir,
        out_dir: a.out_dir,
        seg: a.seg,
        batch: a.batch,
        steps: a.steps,
        lr: a.lr,
        val_every: a.val_every,
        seed: a.seed,
        checkpoint_every: a.checkpoint_every,
        nsf: a.nsf,
    };
    println!("== voice train (device {}) ==", a.device);
    let s = train::run_training(&train_ds, &val_ds, &cfg, &device)?;

    let ratio = s.first / s.last;
    println!(
        "\ntrain mel-L1: {:.5} → {:.5} (giảm {:.2}×)",
        s.first, s.last, ratio
    );
    println!("val mel-L1 tốt nhất: {:.5}", s.best_val);
    println!("adapter cuối: {}", s.adapter_path.display());
    println!("adapter tốt nhất: {}", s.best_path.display());
    println!("lịch sử: {}", s.history_path.display());
    if let Some(cp) = &s.checkpoint_path {
        println!("checkpoint: {}", cp.display());
    }

    // Đẩy kết quả (adapter tốt nhất + cuối + lịch sử + checkpoint nếu có).
    if let Some(hub) = &hub {
        let mut files = vec![
            s.best_path.clone(),
            s.adapter_path.clone(),
            s.history_path.clone(),
        ];
        if let Some(cp) = &s.checkpoint_path {
            files.push(cp.clone());
        }
        println!("== đẩy HuggingFace Hub ({}) ==", hub.repo);
        hub.push(&files)?;
    }

    if !(ratio.is_finite() && ratio > 1.0) {
        eprintln!("⚠️  loss KHÔNG giảm — kiểm dữ liệu/lr (chưa tính là FAIL, chạy `eval`)");
    }
    Ok(ExitCode::SUCCESS)
}

fn run_generate(a: GenerateArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let device = device::select(&a.device)?;
    let cfg = render::RenderConfig {
        dec_dir: a.dec_dir,
        feats: a.feats,
        adapter: a.adapter,
        out_wav: a.out_wav,
        seg: a.seg,
    };
    println!("== voice generate (device {}) ==", a.device);
    let s = render::render(&cfg, &device)?;
    println!(
        "sinh {} frame latent = {} mẫu = {:.2} s → {}",
        s.t_out,
        s.samples,
        s.seconds,
        s.out_wav.display()
    );
    Ok(ExitCode::SUCCESS)
}

fn run_validate(a: EvalArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let device = device::select(&a.device)?;

    let entries = manifest::read(&a.val_manifest)?;
    let ds = Dataset::load(&entries)?;
    let adapter = Adapter::load(&a.adapter, &device)?;
    let gen = Generator::load(&a.dec_dir, &device)?;
    let mel_cfg = MelConfig::ljs();
    let basis = mel::mel_basis(&mel_cfg, &device);

    // Đo drift: hai lượt độc lập phải cho cùng mel-L1 (tất định).
    let r1 = eval::evaluate(&adapter, &gen, &ds, &mel_cfg, &basis, &device, &device)?;
    let r2 = eval::evaluate(&adapter, &gen, &ds, &mel_cfg, &basis, &device, &device)?;
    let drift = (r1.mean_mel_l1 - r2.mean_mel_l1).abs();
    let report = eval::report_from(r1.per_utt.clone(), drift);

    // Ngưỡng: mặc định → file gate → override CLI.
    let mut gate = match &a.gate {
        Some(p) => eval::Gate::from_file(p)?,
        None => eval::Gate::default(),
    };
    if let Some(v) = a.max_mel_l1 {
        gate.max_mel_l1 = v;
    }
    if let Some(v) = a.max_speaker_gap {
        gate.max_speaker_gap = v;
    }
    if let Some(v) = a.min_rms {
        gate.min_rms = v;
    }
    if let Some(v) = a.max_rms {
        gate.max_rms = v;
    }
    if let Some(v) = a.max_drift {
        gate.max_drift = v;
    }

    print_report(&report);

    // Tuỳ chọn: báo cáo CSV.
    if let Some(path) = &a.out_report {
        let mut s = String::from("utt_id,speaker_id,mel_l1,rms,samples,finite\n");
        for m in &report.per_utt {
            s.push_str(&format!(
                "{},{},{:.6},{:.6},{},{}\n",
                m.utt_id, m.speaker_id, m.mel_l1, m.rms, m.samples, m.finite
            ));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, s)?;
        println!("đã ghi báo cáo: {}", path.display());
    }

    // Tuỳ chọn: ghi audio sinh (nghe thử).
    if let Some(dir) = &a.out_audio_dir {
        std::fs::create_dir_all(dir)?;
        for (u, utt) in ds.utts.iter().enumerate() {
            let t_out = ds.latent_len(u).max(1);
            let t_h = Dataset::hubert_frames_for(t_out).min(utt.t_h).max(1);
            let host = ds.full_feats_channel_major(u, t_out);
            let c = utt.c;
            let feats = burn::tensor::Tensor::<3>::from_data(
                burn::tensor::TensorData::new(host, [1, c, t_h]),
                &device,
            );
            let z = adapter.forward_to(feats, t_out);
            let wav = gen
                .forward(z, None, false)
                .reshape([1, t_out * voice_train::dataset::SAMPLES_PER_FRAME])
                .into_data()
                .as_slice::<f32>()?
                .to_vec();
            let path = dir.join(format!("{}.wav", utt.utt_id));
            voice_train::data::write_wav_mono(&path, &wav, 22050)?;
        }
        println!("đã ghi audio vào {}", dir.display());
    }

    let (pass, fails) = gate.check(&report);
    let mut fails = fails;

    // Baseline: so với summary.json của version trước (golden).
    let mut baseline_val: Option<f64> = None;
    if let Some(path) = &a.baseline {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("mở baseline {} lỗi: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("baseline {} không phải JSON: {e}", path.display()))?;
        let b = v
            .get("mean_mel_l1")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| format!("baseline {} thiếu `mean_mel_l1`", path.display()))?;
        baseline_val = Some(b);
        match eval::regression_fail(report.mean_mel_l1, b, a.max_regression) {
            Some(msg) => fails.push(msg),
            None => println!(
                "so baseline: mel_l1 {:.5} ≤ {:.5} (ok)",
                report.mean_mel_l1, b
            ),
        }
    }

    let pass = pass && fails.is_empty();

    // Tóm tắt JSON (golden cho lần sau; cả khi FAIL để còn soi).
    if let Some(path) = &a.out_json {
        let summary = serde_json::json!({
            "mean_mel_l1": report.mean_mel_l1,
            "speaker_gap": report.speaker_gap,
            "min_rms": report.min_rms,
            "max_rms": report.max_rms,
            "drift": report.drift,
            "all_finite": report.all_finite,
            "n_utt": report.per_utt.len(),
            "n_speakers": report.per_speaker.len(),
            "baseline_mel_l1": baseline_val,
            "max_regression": a.max_regression,
            "pass": pass,
            "fails": fails,
        });
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
        println!("đã ghi tóm tắt: {}", path.display());
    }

    println!();
    if pass {
        println!("✅ PASS — đạt cổng chất lượng");
        Ok(ExitCode::SUCCESS)
    } else {
        println!("❌ FAIL — vi phạm cổng:");
        for f in &fails {
            println!("   • {f}");
        }
        Ok(ExitCode::from(1))
    }
}

fn run_diag(a: DiagArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let device = voice_train::device::select("cpu")?;
    let (sum, rows) = voice_train::diag::diagnose(
        &a.target,
        &a.gen,
        a.f0.as_deref(),
        &device,
    )?;
    sum.print();
    if let Some(path) = &a.out_csv {
        voice_train::diag::write_csv(path, &rows, a.f0.is_some())?;
        println!("đã ghi {} frame → {}", rows.len(), path.display());
    }
    // Heuristic đọc nhanh: voiced lệch nhiều hơn unvoiced → nghi thiếu excitation (NSF);
    // mọi band + unvoiced đều lệch → z ngoài manifold (bridge).
    match (sum.voiced, sum.unvoiced) {
        (Some(v), Some(u)) if v > 2.0 * u && v > 0.3 => {
            println!("\n🔑 voiced ≫ unvoiced — nghi THIẾU EXCITATION → NSF là hướng đúng (§13.4.1)");
        }
        (Some(_), Some(u)) if u > 0.3 && sum.bands.iter().all(|(_, x)| *x > 0.3) => {
            println!("\n🔑 mọi band + unvoiced đều lệch — nghi Z NGOÀI MANIFOLD (bridge) — NSF không phải nút thắt chính");
        }
        _ => {} // không đủ dữ liệu để quyết — đọc bảng manual
    }
    Ok(ExitCode::SUCCESS)
}

fn run_verify(a: VerifyArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let dir = PathBuf::from(&a.dec_dir);
    let audio_npz = dir.join("ref_audio.npz");
    let input_npz = dir.join("ref_input.npz");
    if !audio_npz.exists() {
        return Err(format!("{audio_npz:?} chưa có — chạy ./scripts/fetch_assets.sh").into());
    }

    let device = device::select(&a.device)?;
    println!("Nạp generator từ {} ...", dir.display());
    let gen = Generator::load(&dir, &device)?;
    println!("  OK\n");

    let ref_a = read_npz(&audio_npz)?;
    let ref_z = read_npz(&input_npz)?;

    let mut keys: Vec<&String> = ref_a.keys().collect();
    keys.sort_by_key(|k| k.trim_start_matches('T').parse::<usize>().unwrap_or(0));

    let mut all_pass = true;
    let mut worst_overall = 0.0f64;

    println!(
        "{:>5}  {:>10}  {:>10}  {:>12}  {:>11}  {}",
        "T", "samples", "mong doi", "max abs diff", "rms", "ket qua"
    );

    for k in &keys {
        let t: usize = k.trim_start_matches('T').parse()?;
        let expect_n = t * 256;
        let (z_flat, _z_dims) = ref_z.get(*k).expect("thieu z");
        let (ref_flat, _) = ref_a.get(*k).expect("thieu audio");

        let z = Tensor::from_data(TensorData::new(z_flat.clone(), [1usize, 192, t]), &device);
        let out = gen.forward(z, None, false);
        let [b, c, n] = out.dims();

        if b != 1 || c != 1 {
            println!("T={t:<5} shape {b}x{c}x{n} — SAI shape, bo qua");
            all_pass = false;
            continue;
        }
        if n != expect_n {
            println!(
                "T={t:<5} {n:>10}  {expect_n:>10}  {:>12}  {:>11}  FAIL chieu dai",
                "-", "-"
            );
            all_pass = false;
            continue;
        }

        let got = out.into_data().as_slice::<f32>()?.to_vec();

        let mut max_abs = 0.0f64;
        let mut sq = 0.0f64;
        let mut sq_ref = 0.0f64;
        for (a, b) in got.iter().zip(ref_flat.iter()) {
            let d = (*a as f64 - *b as f64).abs();
            if d > max_abs {
                max_abs = d;
            }
            sq += (*a as f64).powi(2);
            sq_ref += (*b as f64).powi(2);
        }
        let rms_got = (sq / got.len() as f64).sqrt();
        let rms_ref = (sq_ref / ref_flat.len() as f64).sqrt();
        worst_overall = worst_overall.max(max_abs);

        let ok = max_abs < 1e-4 && (rms_got - rms_ref).abs() < 1e-5;
        all_pass &= ok;
        println!(
            "{t:>5}  {n:>10}  {expect_n:>10}  {max_abs:>12.3e}  {rms_got:>11.6}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    println!("\nmax abs diff lớn nhất: {worst_overall:.3e}");
    {
        let (data, _) = ref_a.values().next().expect("khong co mau nao");
        let sq: f64 = data.iter().map(|v| (*v as f64).powi(2)).sum();
        println!(
            "rms chuỗi đầu (PyTorch): {:.6}   n = {}",
            (sq / data.len() as f64).sqrt(),
            data.len()
        );
    }
    println!("\n{}", if all_pass { "PASS tat ca" } else { "CO FAIL" });
    if !all_pass {
        std::process::exit(1);
    }
    Ok(ExitCode::SUCCESS)
}

fn print_report(r: &eval::EvalReport) {
    println!("\n== voice validate ==");
    println!("câu: {} · giọng: {}", r.per_utt.len(), r.per_speaker.len());
    println!("mel-L1 trung bình  : {:.5}", r.mean_mel_l1);
    println!("speaker_gap        : {:.4}", r.speaker_gap);
    println!(
        "rms [min, max]     : [{:.3e}, {:.3e}]",
        r.min_rms, r.max_rms
    );
    println!("drift              : {:.3e}", r.drift);
    println!("speaker-similarity : N/A (cần ContentVec so sánh — Luồng 2)");
    if !r.all_finite {
        println!("⚠️  có mẫu audio không hữu hạn");
    }
    println!("\nmel-L1 theo giọng:");
    for (spk, s) in &r.per_speaker {
        println!("  {spk:>12}  {:>3} câu  {:.5}", s.count, s.mean_mel_l1);
    }
}
