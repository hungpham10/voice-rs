//! `voice train` — Luồng 1: cầu nối nội dung nhiều giọng (§13.6.4f).
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
use voice_lib::vc::{Adapter, nearest_index_map};
use voice_mel::MelConfig;
use voice_vits::dec::Generator;

use crate::dataset::{Dataset, Rng, HUBERT_HZ, LATENT_HZ, SAMPLES_PER_FRAME};
use crate::{eval, mel};

/// Đáy đoạn huấn luyện (frame latent). STFT `n_fft=1024, center=true` cần tín hiệu
/// > n_fft/2 = 512 mẫu → cần ≥ 3 frame; lấy 32 frame (~0,37 s) để loss mel ổn định.
const MIN_SEG: usize = 32;

/// Cấu hình `voice train`.
#[derive(Debug, Clone)]
pub struct TrainConfig {
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
    /// `0` = tắt; `>0` = cứ mỗi N bước lưu `checkpoint.bpk` trong `out_dir`.
    pub checkpoint_every: usize,
    /// Bật NSF-lite: unfreeze `conv_src` và bơm excitation từ f0 sau `conv_pre`.
    /// Yêu cầu bundle dec có `conv_src.npy` và mỗi câu có `.f0.npy` co-location.
    pub nsf: bool,
}

/// Kết quả huấn luyện.
#[derive(Debug, Clone)]
pub struct TrainingSummary {
    pub first: f64,
    pub last: f64,
    pub best_val: f64,
    pub seg: usize,
    pub batch: usize,
    pub params: usize,
    pub adapter_path: PathBuf,
    pub best_path: PathBuf,
    pub history_path: PathBuf,
    /// File checkpoint định kỳ (nếu `checkpoint_every > 0`).
    pub checkpoint_path: Option<PathBuf>,
}

/// Huấn luyện cầu nối. `device` là CPU/… trước autodiff (hàm tự bật autodiff).
pub fn run_training(
    train_ds: &Dataset,
    val_ds: &Dataset,
    cfg: &TrainConfig,
    device: &Device,
) -> Result<TrainingSummary, Box<dyn std::error::Error>> {
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
    let seg = if cfg.seg == 0 {
        // Cả câu: lấy dài nhất có thể theo câu ngắn nhất, nhưng không dưới đáy MIN_SEG.
        min_latent.max(MIN_SEG)
    } else {
        cfg.seg.max(MIN_SEG).min(min_latent)
    };
    let t_h_seg = Dataset::hubert_frames_for(seg);
    let n = seg * SAMPLES_PER_FRAME;

    // Lọc các câu quá ngắn (< seg frame latent hoặc < t_h_seg frame HuBERT) ra khỏi
    // lấy mẫu để tránh underflow / crash STFT trên STFT reflect-pad.
    let ineligible: Vec<usize> = (0..train_ds.utts.len())
        .filter(|&u| {
            let lat = train_ds.latent_len(u);
            let t_h = train_ds.utts[u].t_h;
            lat < seg || t_h < t_h_seg
        })
        .collect();
    let ineligible_cnt = ineligible.len();
    if ineligible_cnt > 0 {
        println!(
            "⚠️  bỏ {ineligible_cnt} câu quá ngắn (< {seg} frame latent) để tránh crash STFT"
        );
    }
    let active: Vec<usize> = (0..train_ds.num_speakers())
        .filter(|&s| {
            train_ds.per_speaker[s]
                .iter()
                .any(|&u| !ineligible.contains(&u))
        })
        .collect();
    if active.is_empty() {
        return Err(format!(
            "không còn câu đủ dài (≥ {seg} frame latent) để train — hãy giảm --seg hoặc bổ sung dữ liệu dài hơn"
        )
        .into());
    }
    let per_spk_active: Vec<Vec<usize>> = train_ds
        .per_speaker
        .iter()
        .map(|list| {
            list.iter()
                .copied()
                .filter(|&u| !ineligible.contains(&u))
                .collect()
        })
        .collect();

    let device = device.clone().autodiff();
    let device_eval = Device::flex(); // CPU, không autodiff → forward eval không dựng graph gradient
    let mut gen = Generator::load(&cfg.dec_dir, &device)?.freeze();
    let gen_eval = Generator::load(&cfg.dec_dir, &device_eval)?.freeze();

    // NSF-lite: unfreeze conv_src (chỉ có khi bundle có conv_src.npy).
    if cfg.nsf {
        let src = gen
            .conv_src
            .take()
            .ok_or_else(|| {
                format!(
                    "voice train --nsf yêu cầu bundle dec có conv_src.npy (thiếu ở {})",
                    cfg.dec_dir.display()
                )
            })?;
        let src = src.unfreeze();
        gen.conv_src = Some(src);
    }

    // Khi bật NSF, mọi câu phải kèm .f0.npy co-location.
    if cfg.nsf {
        for ds in [train_ds, val_ds] {
            for utt in &ds.utts {
                if utt.f0.is_none() {
                    return Err(format!(
                        "voice train --nsf: câu {} thiếu features .f0.npy co-location",
                        utt.utt_id
                    )
                    .into());
                }
            }
        }
    }

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
        "seg {seg} frame latent ({t_h_seg} frame HuBERT, {n} mẫu) · batch {} · {} giọng/bước",
        cfg.batch,
        active.len()
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

    let checkpoint_path = if cfg.checkpoint_every > 0 {
        Some(cfg.out_dir.join("checkpoint.bpk"))
    } else {
        None
    };

    for step in 0..cfg.steps {
        let mut feats_host: Vec<f32> = Vec::with_capacity(cfg.batch * c * t_h_seg);
        let mut wav_host: Vec<f32> = Vec::with_capacity(cfg.batch * n);
        let mut f0_host: Vec<f32> = vec![0.0f32; cfg.batch * seg];

        for i in 0..cfg.batch {
            // Round-robin theo giọng → batch cân bằng bất kể số câu mỗi giọng.
            let spk = active[cursor % active.len()];
            cursor = (cursor + 1) % active.len();
            let list = &per_spk_active[spk];
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

            // f0 của câu này, resample từ 100 Hz (RMVPE) về seg frame latent (86.13 Hz)
            if cfg.nsf {
                let f0 = train_ds.utts[u].f0.as_ref().expect("f0 đã kiểm ở trên");
                let map = nearest_index_map(f0.len(), seg);
                for t in 0..seg {
                    f0_host[i * seg + t] = f0[map[t]];
                }
            }
        }

        let b = cfg.batch;
        let feats = Tensor::<3>::from_data(TensorData::new(feats_host, [b, c, t_h_seg]), &device);
        let target = Tensor::<2>::from_data(TensorData::new(wav_host, [b, n]), &device);
        let f0_tensor = if cfg.nsf {
            Some(Tensor::<2>::from_data(
                TensorData::new(f0_host, [b, seg]),
                &device,
            ))
        } else {
            None
        };

        let z = adapter.forward_to(feats, seg);
        let wav = gen.forward(z, f0_tensor, false).reshape([b, n]);
        let loss = mel::mel_l1(wav, target, &mel_cfg, &basis);

        let lv: f64 = loss.clone().into_scalar::<f32>() as f64;
        if step == 0 {
            first = lv;
        }
        last = lv;

        let grads = GradientsParams::from_grads(loss.backward(), &adapter);
        adapter = optim.step(cfg.lr, adapter, grads);

        if let Some(path) = &checkpoint_path {
            if (step + 1) % cfg.checkpoint_every == 0 {
                adapter.save(path)?;
                println!("  ↳ checkpoint step {} → {}", step + 1, path.display());
            }
        }

        let do_val =
            cfg.val_every > 0 && ((step + 1) % cfg.val_every == 0 || step + 1 == cfg.steps);
        let val_now = if do_val {
            let report = eval::evaluate(&adapter, &gen_eval, val_ds, &mel_cfg, &basis, &device, &device_eval)?;
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

    Ok(TrainingSummary {
        first,
        last,
        best_val,
        seg,
        batch: cfg.batch,
        params,
        adapter_path,
        best_path,
        history_path,
        checkpoint_path,
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;
    use crate::dataset::{Dataset, Utterance};
    use crate::eval::per_utt_metrics;
    use crate::mel;
    use voice_lib::vc::Adapter;
    use voice_mel::MelConfig;
    use voice_vits::dec::Generator;
    use std::path::{Path, PathBuf};

    fn make_utt(speaker: usize, utt_id: &str, t_h: usize) -> Utterance {
        Utterance {
            speaker,
            utt_id: utt_id.to_string(),
            t_h,
            c: 768,
            feats: vec![0.0f32; t_h * 768],
            wav: vec![0.1f32; t_h * 256],
            f0: None,
        }
    }

    fn make_ds() -> Dataset {
        // Một câu cực ngắn (1 frame latent = 256 mẫu) — đây là trigger của bug STFT
        // "signal length (256) must be > n_fft/2 (512)".
        let mut utts = vec![make_utt(0, "short", 1)];
        for i in 0..5usize {
            utts.push(make_utt(0, format!("long{i}").as_str(), 40));
        }
        let mut per_speaker = vec![Vec::new(); 1];
        for (i, u) in utts.iter().enumerate() {
            per_speaker[u.speaker].push(i);
        }
        Dataset {
            speakers: vec!["spk0".to_string()],
            utts,
            per_speaker,
        }
    }

    #[test]
    fn seg_zero_does_not_panic_on_short_utterance() {
        if !Path::new("sources/vits_ljs_dec/config.json").exists() {
            eprintln!("skip: sources/vits_ljs_dec thiếu");
            return;
        }
        let device = Device::flex();
        let train_ds = make_ds();
        let val_ds = make_ds();

        let cfg = TrainConfig {
            dec_dir: PathBuf::from("sources/vits_ljs_dec"),
            out_dir: std::env::temp_dir().join("voice_rs_seg_test_train"),
            seg: 0,
            batch: 2,
            steps: 5,
            lr: 1e-3,
            val_every: 5,
            seed: 42,
            checkpoint_every: 0,
            nsf: false,
        };
        let summary = run_training(&train_ds, &val_ds, &cfg, &device).expect("train không panic");
        assert!(summary.first.is_finite());
        assert!(summary.last.is_finite());
        assert_eq!(summary.seg, MIN_SEG, "seg=0 với câu 1-frame phải bám đáy MIN_SEG");
    }

    #[test]
    fn eval_skips_short_utterances() {
        if !Path::new("sources/vits_ljs_dec/config.json").exists() {
            eprintln!("skip: sources/vits_ljs_dec thiếu");
            return;
        }
        let device = Device::flex();
        let ds = make_ds();
        let gen = Generator::load(&PathBuf::from("sources/vits_ljs_dec"), &device)
            .expect("nạp dec")
            .freeze();
        let mel_cfg = MelConfig::ljs();
        let basis = mel::mel_basis(&mel_cfg, &device);
        let adapter = Adapter::new(&device).unfreeze();

        let metrics = per_utt_metrics(&adapter, &gen, &ds, &mel_cfg, &basis, &device, &device)
            .expect("per_utt_metrics không panic");
        // 6 câu total, 1 câu short bị bỏ → 5 câu có metric
        assert_eq!(metrics.len(), 5, "câu 1-frame phải bị bỏ qua trong eval");
        assert!(metrics.iter().all(|m| m.mel_l1.is_finite()));
    }
}
