//! `train bridge` — Luồng 1: cầu nối nội dung nhiều giọng (§13.6.4f).
//!
//! Chỉ `Adapter` (`Linear(768→192)`) học; `dec` **đóng băng**. Mỗi bước lấy một
//! batch **cân bằng theo giọng** (round-robin), mỗi giọng một câu ngẫu nhiên, cắt
//! một đoạn latent ngẫu nhiên. Loss = `L1` log-mel ở 22.05 kHz.
//!
//! Mọi nguồn ngẫu nhiên đi qua [`Rng`] với seed cố định (không dùng `rand` để
//! không thêm phụ thuộc và để seed tái lập được trên mọi nền tảng).

use std::path::PathBuf;

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Tensor, TensorData};
use voice_lib::vc::Adapter;
use voice_mel::MelConfig;
use voice_models::dec::Generator;

use crate::dataset::{Dataset, Rng, HUBERT_HZ, LATENT_HZ, SAMPLES_PER_FRAME};
use crate::{eval, mel};

/// Cấu hình `train bridge`.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub dec_dir: PathBuf,
    pub out_dir: PathBuf,
    /// Số frame latent mỗi mẫu (tự co theo câu ngắn nhất).
    pub seg: usize,
    /// Số câu mỗi bước.
    pub batch: usize,
    pub steps: usize,
    pub lr: f64,
    /// Cứ bao nhiêu bước đo val một lần.
    pub val_every: usize,
    pub seed: u64,
}

/// Kết quả huấn luyện.
#[derive(Debug, Clone)]
pub struct BridgeSummary {
    pub first: f64,
    pub last: f64,
    pub best_val: f64,
    pub seg: usize,
    pub batch: usize,
    pub params: usize,
    pub adapter_path: PathBuf,
    pub best_path: PathBuf,
    pub history_path: PathBuf,
}

/// Huấn luyện cầu nối. `device` là CPU/… trước autodiff (hàm tự bật autodiff).
pub fn train_bridge(
    train_ds: &Dataset,
    val_ds: &Dataset,
    cfg: &BridgeConfig,
    device: &Device,
) -> Result<BridgeSummary, Box<dyn std::error::Error>> {
    let c = train_ds.check_dims()?;
    val_ds.check_dims()?;
    if val_ds.utts.is_empty() {
        return Err("val rỗng — cần ít nhất một câu để đo".into());
    }
    let nspk = train_ds.num_speakers();
    if nspk == 0 {
        return Err("train rỗng".into());
    }

    // `seg` hiệu dụng: không vượt câu ngắn nhất (để batch đều kích thước).
    let min_latent = (0..train_ds.utts.len())
        .map(|u| train_ds.latent_len(u))
        .min()
        .unwrap_or(1)
        .max(1);
    let seg = cfg.seg.min(min_latent).max(1);
    let t_h_seg = Dataset::hubert_frames_for(seg);
    let n = seg * SAMPLES_PER_FRAME;

    let device = device.clone().autodiff();
    let gen = Generator::load(&cfg.dec_dir, &device)?.freeze();
    let mut adapter = Adapter::new(&device).unfreeze();
    let params = adapter.in_dim() * adapter.out_dim() + adapter.out_dim();

    println!(
        "dec (đóng băng) = {:.2} M tham số · adapter (học) = {params} tham số",
        gen.num_params() as f64 / 1e6
    );
    println!(
        "{} giọng · {} câu train · {} câu val",
        nspk,
        train_ds.utts.len(),
        val_ds.utts.len()
    );
    println!(
        "seg {seg} frame latent ({t_h_seg} frame HuBERT, {n} mẫu) · batch {} · {nspk} giọng/bước",
        cfg.batch
    );

    let mel_cfg = MelConfig::ljs();
    let basis = mel::mel_basis(&mel_cfg, &device);

    let mut optim = AdamConfig::new().init();
    let mut rng = Rng::new(cfg.seed);
    let mut cursor = 0usize;

    std::fs::create_dir_all(&cfg.out_dir)?;
    let adapter_path = cfg.out_dir.join("adapter.bpk");
    let best_path = cfg.out_dir.join("adapter_best.bpk");
    let history_path = cfg.out_dir.join("history.csv");

    let mut first = f64::NAN;
    let mut last = f64::NAN;
    let mut best_val = f64::INFINITY;
    let mut best_adapter = adapter.clone();
    // (step, train_mel_l1, val_mel_l1) — val chỉ có ở bước đo.
    let mut history: Vec<(usize, f64, Option<f64>)> = Vec::with_capacity(cfg.steps);

    for step in 0..cfg.steps {
        let mut feats_host: Vec<f32> = Vec::with_capacity(cfg.batch * c * t_h_seg);
        let mut wav_host: Vec<f32> = Vec::with_capacity(cfg.batch * n);

        for _ in 0..cfg.batch {
            // Round-robin theo giọng → batch cân bằng bất kể số câu mỗi giọng.
            let spk = cursor % nspk;
            cursor = (cursor + 1) % nspk;
            let list = &train_ds.per_speaker[spk];
            let u = list[rng.below(list.len())];
            let lat = train_ds.latent_len(u);

            let l0 = rng.below(lat - seg + 1);
            let t_h = train_ds.utts[u].t_h;
            let mut h0 = (l0 as f64 * HUBERT_HZ / LATENT_HZ).floor() as usize;
            if h0 + t_h_seg > t_h {
                h0 = t_h - t_h_seg;
            }

            feats_host.extend_from_slice(&train_ds.segment_feats_channel_major(u, h0, t_h_seg));
            let (wav_seg, _) = train_ds.segment_wav(u, l0, seg);
            wav_host.extend_from_slice(&wav_seg);
        }

        let b = cfg.batch;
        let feats = Tensor::<3>::from_data(TensorData::new(feats_host, [b, c, t_h_seg]), &device);
        let target = Tensor::<2>::from_data(TensorData::new(wav_host, [b, n]), &device);

        let z = adapter.forward_to(feats, seg);
        let wav = gen.forward(z).reshape([b, n]);
        let loss = mel::mel_l1(wav, target, &mel_cfg, &basis);

        let lv: f64 = loss.clone().into_scalar::<f32>() as f64;
        if step == 0 {
            first = lv;
        }
        last = lv;

        let grads = GradientsParams::from_grads(loss.backward(), &adapter);
        adapter = optim.step(cfg.lr, adapter, grads);

        let do_val = cfg.val_every > 0 && ((step + 1) % cfg.val_every == 0 || step + 1 == cfg.steps);
        let val_now = if do_val {
            let report = eval::evaluate(&adapter, &gen, val_ds, &mel_cfg, &basis, &device)?;
            let improved = report.mean_mel_l1 < best_val;
            if improved {
                best_val = report.mean_mel_l1;
                best_adapter = adapter.clone();
            }
            println!(
                "step {:>4}  train mel-L1 = {lv:.5}  val = {:.5}{}",
                step + 1,
                report.mean_mel_l1,
                if improved { "  ← tốt nhất" } else { "" }
            );
            Some(report.mean_mel_l1)
        } else {
            if step % 25 == 0 || step + 1 == cfg.steps {
                println!("step {:>4}  train mel-L1 = {lv:.5}", step + 1);
            }
            None
        };

        history.push((step + 1, lv, val_now));
        write_history(&history_path, &history)?;
    }

    adapter.save(&adapter_path)?;
    best_adapter.save(&best_path)?;

    Ok(BridgeSummary {
        first,
        last,
        best_val,
        seg,
        batch: cfg.batch,
        params,
        adapter_path,
        best_path,
        history_path,
    })
}

/// Ghi lịch sử huấn luyện (ghi lại toàn bộ mỗi lần gọi — đủ nhỏ để đơn giản).
fn write_history(
    path: &std::path::Path,
    rows: &[(usize, f64, Option<f64>)],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut s = String::from("step,train_mel_l1,val_mel_l1\n");
    for (step, train, val) in rows {
        match val {
            Some(v) => s.push_str(&format!("{step},{train:.6},{v:.6}\n")),
            None => s.push_str(&format!("{step},{train:.6},\n")),
        }
    }
    std::fs::write(path, s)?;
    Ok(())
}
