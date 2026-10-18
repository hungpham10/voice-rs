//! CLI hợp nhất cho đường huấn luyện VC 1→1 (§13.6.4a).
//!
//! ```text
//! voice-train train overfit --wav ... --feats ... [--device cpu]
//! voice-train train bridge  --train-manifest ... --val-manifest ... [--device cpu]
//! voice-train render        --feats ... --adapter ... [--out-wav render.wav]
//! voice-train eval          --val-manifest ... --adapter ... [--gate gate.toml]
//! ```
//!
//! - `train overfit` = GĐ4 (overfit 1 câu, integration test) — trước là bin
//!   `train_adapter`.
//! - `train bridge` = Luồng 1 (cầu nối nội dung nhiều giọng).
//! - `render` = **suy diễn thuần** feats + adapter → wav (mặc định cả câu).
//! - `eval` thoát với mã **1** khi vi phạm cổng — dùng trực tiếp trong CI/Colab.
//!
//! Cả hai lệnh `train` đều hỗ trợ **checkpoint định kỳ** (`--checkpoint-every`)
//! và **đẩy kết quả lên HuggingFace Hub** (`--push-to-hub`, mặc định tắt).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use voice_lib::vc::Adapter;
use voice_mel::MelConfig;
use voice_models::dec::Generator;
use voice_train::dataset::Dataset;
use voice_train::hub::HubConfig;
use voice_train::overfit::OverfitConfig;
use voice_train::train::BridgeConfig;
use voice_train::{device, eval, manifest, mel, overfit, render, train};

#[derive(Parser)]
#[command(
    name = "voice-train",
    version,
    about = "Huấn luyện + đánh giá adapter VC 1→1"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Huấn luyện
    Train {
        #[command(subcommand)]
        what: TrainCmd,
    },
    /// Suy diễn: feats + adapter → wav (không huấn luyện; mặc định cả câu)
    Render(RenderArgs),
    /// Đánh giá + cổng PASS/FAIL (mã thoát 1 khi FAIL)
    Eval(EvalArgs),
    /// Chẩn đoán rè/nhiễu: mel-L1 theo band + voiced/unvoiced (PR-B1)
    Diag(DiagArgs),
}

#[derive(Subcommand)]
enum TrainCmd {
    /// GĐ4 — overfit 1 câu (integration test; chỉ `Adapter` học, `dec` đóng băng)
    Overfit(OverfitArgs),
    /// Luồng 1 — cầu nối nội dung nhiều giọng (chỉ `Adapter` học, `dec` đóng băng)
    Bridge(BridgeArgs),
}

#[derive(Args)]
struct RenderArgs {
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
    /// `cpu` | `cuda` | `metal` (mặc định cpu).
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

#[derive(Args)]
struct OverfitArgs {
    #[arg(long, default_value = "sources/vits_ljs_dec")]
    dec_dir: PathBuf,
    /// Wav giọng đích (16 kHz mono).
    #[arg(long, default_value = "sources/422-122949-0031.wav")]
    wav: PathBuf,
    /// Features ContentVec (`.npy`) của wav đích.
    #[arg(long, default_value = "sources/features/422-122949-0031.npy")]
    feats: PathBuf,
    #[arg(long, default_value = "test_output.wav")]
    out_wav: PathBuf,
    #[arg(long, default_value = "sources/features/adapter.bpk")]
    out_adapter: PathBuf,
    /// Số frame latent; `0` (mặc định) = **cả câu**, tự co theo dữ liệu thật.
    #[arg(long, default_value_t = 0)]
    seg: usize,
    #[arg(long, default_value_t = 300)]
    steps: usize,
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    /// `0` = tắt; `>0` = cứ mỗi N bước lưu `checkpoint.bpk` cạnh `--out-adapter`.
    #[arg(long, default_value_t = 0)]
    checkpoint_every: usize,
    /// `cpu` | `cuda` | `metal` (mặc định cpu).
    #[arg(long, default_value = "cpu")]
    device: String,
    #[command(flatten)]
    hub: HubArgs,
}

#[derive(Args)]
struct BridgeArgs {
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
    /// `cpu` | `cuda` | `metal` (mặc định cpu).
    #[arg(long, default_value = "cpu")]
    device: String,
    #[command(flatten)]
    hub: HubArgs,
}

#[derive(Parser)]
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

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Train { what } => match what {
            TrainCmd::Overfit(a) => run_overfit(a),
            TrainCmd::Bridge(a) => run_bridge(a),
        },
        Cmd::Render(a) => run_render(a),
        Cmd::Eval(a) => run_eval(a),
        Cmd::Diag(a) => run_diag(a),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("❌ {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_overfit(a: OverfitArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let hub = a.hub.resolve()?;
    let device = device::select(&a.device)?;

    let cfg = OverfitConfig {
        dec_dir: a.dec_dir,
        wav: a.wav,
        feats: a.feats,
        out_wav: a.out_wav.clone(),
        out_adapter: a.out_adapter.clone(),
        seg: a.seg,
        steps: a.steps,
        lr: a.lr,
        checkpoint_every: a.checkpoint_every,
    };
    println!("== train overfit (device {}) ==", a.device);
    let s = overfit::train_overfit(&cfg, &device)?;

    // Đẩy kết quả (adapter cuối + checkpoint nếu có).
    if let Some(hub) = &hub {
        let mut files = vec![s.out_adapter.clone()];
        if let Some(cp) = &s.checkpoint_path {
            files.push(cp.clone());
        }
        println!("== đẩy HuggingFace Hub ({}) ==", hub.repo);
        hub.push(&files)?;
    }

    if !(s.ratio.is_finite() && s.ratio > 1.0) {
        eprintln!("\n❌ loss KHÔNG giảm — kiểm dữ liệu/lr");
        return Ok(ExitCode::from(1));
    }
    println!("\nPASS — loss giảm {:.2}×", s.ratio);
    Ok(ExitCode::SUCCESS)
}

fn run_bridge(a: BridgeArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
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

    let cfg = BridgeConfig {
        dec_dir: a.dec_dir,
        out_dir: a.out_dir,
        seg: a.seg,
        batch: a.batch,
        steps: a.steps,
        lr: a.lr,
        val_every: a.val_every,
        seed: a.seed,
        checkpoint_every: a.checkpoint_every,
    };
    println!("== train bridge (device {}) ==", a.device);
    let s = train::train_bridge(&train_ds, &val_ds, &cfg, &device)?;

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

fn run_render(a: RenderArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let device = device::select(&a.device)?;
    let cfg = render::RenderConfig {
        dec_dir: a.dec_dir,
        feats: a.feats,
        adapter: a.adapter,
        out_wav: a.out_wav,
        seg: a.seg,
    };
    println!("== render (device {}) ==", a.device);
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

fn run_eval(a: EvalArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let device = device::select(&a.device)?;

    let entries = manifest::read(&a.val_manifest)?;
    let ds = Dataset::load(&entries)?;
    let adapter = Adapter::load(&a.adapter, &device)?;
    let gen = Generator::load(&a.dec_dir, &device)?;
    let mel_cfg = MelConfig::ljs();
    let basis = mel::mel_basis(&mel_cfg, &device);

    // Đo drift: hai lượt độc lập phải cho cùng mel-L1 (tất định).
    let r1 = eval::evaluate(&adapter, &gen, &ds, &mel_cfg, &basis, &device)?;
    let r2 = eval::evaluate(&adapter, &gen, &ds, &mel_cfg, &basis, &device)?;
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
                .forward(z)
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

fn print_report(r: &eval::EvalReport) {
    println!("\n== eval ==");
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
