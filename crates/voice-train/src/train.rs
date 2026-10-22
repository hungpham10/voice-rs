//! `voice train` — Luồng 1: cầu nối nội dung nhiều giọng (§13.6.4f).
//!
//! Chỉ `Adapter` (`Linear(768→192)`) học; `dec` **đóng băng**. Mỗi bước lấy một
//! batch **cân bằng theo giọng** (round-robin), mỗi giọng một **chunk** ngẫu nhiên
//! (chunk = đoạn ≤ chunk_frames frame latent, cắt ở vùng yên).
//! Loss = `L1` log-mel ở 22.05 kHz.
//!
//! Mọi nguồn ngẫu nhiên đi qua [`Rng`] với seed cố định (không dùng `rand` để
//! không thêm phụ thuộc và để seed tái lập được trên mọi nền tảng).

use std::path::PathBuf;

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Int, Tensor, TensorData};
use voice_lib::vc::{Adapter, nearest_index_map};
use voice_mel::MelConfig;
use voice_vits::dec::Generator;

use crate::dataset::{Dataset, Rng, LATENT_HZ, SAMPLES_PER_FRAME};
use crate::{eval, mel};

/// Đáy đoạn huấn luyện (frame latent). STFT `n_fft=1024, center=true` cần tín hiệu
/// > n_fft/2 = 512 mẫu → cần ≥ 3 frame; lấy 32 frame (~0,37 s) để loss mel ổn định.
const MIN_SEG: usize = 32;

/// Cấu hình `voice train`.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub dec_dir: PathBuf,
    pub out_dir: PathBuf,
    /// Số frame latent mỗi chunk (mỗi sample = 1 chunk). `0` = không chunk (backward compat, không khuyến khích).
    pub chunk_frames: usize,
    /// Số chunk mỗi micro-batch (batch size thật tương ứng memory).
    pub batch: usize,
    /// Cộng dồn gradient qua N micro-batch trước 1 optimizer step.
    /// Effective batch = `batch × grad_accum`; memory chỉ theo `batch` một mình.
    pub grad_accum: usize,
    pub steps: usize,
    pub lr: f64,
    /// Các bước đầu lr tuyến tính 0 → lr (chuẩn Adam để tránh blowup bước đầu).
    pub warmup: usize,
    /// Cắt gradient toàn cục nếu L2 norm vượt quá này (0 = tắt).
    pub clip_norm: f64,
    /// Cứ bao nhiêu bước đo val một lần.
    pub val_every: usize,
    pub seed: u64,
    /// `0` = tắt; `>0` = cứ mỗi N bước lưu `checkpoint.bpk` trong `out_dir`.
    pub checkpoint_every: usize,
    /// Bật NSF-lite: unfreeze `conv_src` và bơm excitation từ f0 sau `conv_pre`.
    /// Yêu cầu bundle dec có `conv_src.npy` và mỗi câu có `.f0.npy` co-location.
    pub nsf: bool,
    /// Phase 3: pitch-embed — f0 quantize 256 bin → Embedding cộng vào z trước
    /// TCN (Applio `emb_pitch`). Yêu cầu `.f0.npy` co-location cho mỗi câu.
    pub pitch_embed: bool,
}

/// Kết quả huấn luyện.
#[derive(Debug, Clone)]
pub struct TrainingSummary {
    pub first: f64,
    pub last: f64,
    pub best_val: f64,
    pub chunk_frames: usize,
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
    if val_ds.chunks.is_empty() {
        return Err("val rỗng — cần ít nhất một chunk để đo".into());
    }
    let nspk = train_ds.num_speakers();
    if nspk == 0 {
        return Err("train rỗng".into());
    }

    // Validate chunk_frames
    let chunk_frames = cfg.chunk_frames;
    if chunk_frames == 0 {
        return Err(
            "chunk_frames = 0 không còn hỗ trợ (v2.2.18+): nạp nguyên câu gây OOM — \
             hãy dùng ≥ MIN_SEG (32), Dataset::load cũng tự chặn"
                .into(),
        );
    }
    if chunk_frames < MIN_SEG {
        return Err(format!("chunk_frames ({chunk_frames}) phải ≥ MIN_SEG ({MIN_SEG})").into());
    }

    let t_h_seg = Dataset::hubert_frames_for(chunk_frames);
    let n = chunk_frames * SAMPLES_PER_FRAME;

    // Active speakers = those with at least 1 chunk
    let active: Vec<usize> = (0..nspk)
        .filter(|&s| !train_ds.per_speaker_chunks[s].is_empty())
        .collect();
    if active.is_empty() {
        return Err("không có giọng nào có chunk hợp lệ để train".into());
    }
    let per_spk_chunks = train_ds.per_speaker_chunks.clone();

    let device = device.clone().autodiff();
    let mut gen = Generator::load(&cfg.dec_dir, &device)?.freeze();

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

    // Khi bật NSF, mọi chunk phải kèm .f0.npy co-location (kiểm tra qua utterance gốc).
    if cfg.nsf {
        for ds in [train_ds, val_ds] {
            for chunk in &ds.chunks {
                let utt = &ds.utts[chunk.utt];
                if utt.f0.is_none() {
                    return Err(format!(
                        "voice train --nsf: chunk {} (utt {}) thiếu features .f0.npy co-location",
                        chunk.chunk_id, utt.utt_id
                    )
                    .into());
                }
            }
        }
    }

    // Pitch-embed vẫn cần f0 cho dù không bật NSF.
    if cfg.pitch_embed && !cfg.nsf {
        for ds in [train_ds, val_ds] {
            for chunk in &ds.chunks {
                let utt = &ds.utts[chunk.utt];
                if utt.f0.is_none() {
                    return Err(format!(
                        "voice train --pitch-embed: chunk {} (utt {}) thiếu features .f0.npy co-location",
                        chunk.chunk_id, utt.utt_id
                    )
                    .into());
                }
            }
        }
    }

    let mut adapter = Adapter::new(&device).unfreeze();
    let params = adapter.in_dim() * adapter.out_dim()
        + adapter.out_dim()
        + voice_lib::vc::PITCH_BINS * adapter.out_dim();

    println!(
        "dec (đóng băng) = {:.2} M tham số · adapter (học) = {params} tham số",
        gen.num_params() as f64 / 1e6
    );
    println!(
        "{} giọng · {} chunk train · {} chunk val",
        nspk,
        train_ds.chunks.len(),
        val_ds.chunks.len()
    );
    // grad accumulation: 1 optimizer step = `accum` micro-batch cộng dồn gradient.
    let accum = cfg.grad_accum.max(1);
    println!(
        "chunk {chunk_frames} frame latent ({t_h_seg} frame HuBERT, {n} mẫu) · batch {}×accum {} (effective {}) · {} giọng/bước",
        cfg.batch,
        accum,
        cfg.batch * accum,
        active.len()
    );

    let mel_cfg = MelConfig::ljs();
    let basis = mel::mel_basis(&mel_cfg, &device);

    let mut optim = AdamConfig::new().init();
    let mut rng = Rng::new(cfg.seed);
    let mut cursor = 0usize;
    // Số optimizer step LIÊN TỤC trong đó mọi micro-batch đều NaN
    // (không reset mỗi bước — phát hiện diverge thật sự, không phải nhiễu).
    let mut consec = 0usize;

    std::fs::create_dir_all(&cfg.out_dir)?;
    let adapter_path = cfg.out_dir.join("adapter.bpk");
    let best_path = cfg.out_dir.join("adapter_best.bpk");
    let history_path = cfg.out_dir.join("history.csv");

    let mut first = f64::NAN;
    let mut last = f64::NAN;
    let mut best_val = f64::INFINITY;
    let mut best_adapter = adapter.clone();
    // best = last ban đầu: nếu val không chạy lần nào, best = bản train cuối.
    let mut val_ran = false;
    let mut history: Vec<(usize, f64, Option<f64>)> = Vec::with_capacity(cfg.steps);

    let checkpoint_path = if cfg.checkpoint_every > 0 {
        Some(cfg.out_dir.join("checkpoint.bpk"))
    } else {
        None
    };

    for step in 0..cfg.steps {
        // ---- Mỗi optimizer step = `accum` micro-batch cộng dồn gradient.
        // Memory chỉ theo `batch` (1 micro-batch), effective batch = batch × accum.
        let mut cum = GradientsParams::new();
        let mut lv_sum = 0f64;
        let mut nan_skips = 0usize; // số micro-batch bị bỏ vì loss NaN
        let mut gn_ema = 0f64;      // EMA chuẩn L2 gradient (phát hiện blowup sớm)
        let mut micro_utts: Vec<(String, usize)> = Vec::with_capacity(cfg.batch); // (utt_id, offset_lat)

        for _micro in 0..accum {
            micro_utts.clear(); // đếm từng micro, không cộng dồn giữa các micro
            // ---- Xếp micro-batch. Chunk có độ dài biến động (plan_chunks Applio:
            // ± chunk/4) → pad 0 về chuẩn `chunk_frames` + mask frame bù loss.
            let mut feats_host: Vec<f32> = vec![0.0f32; cfg.batch * c * t_h_seg];
            let mut wav_host: Vec<f32> = vec![0.0f32; cfg.batch * n];
            let mut f0_host: Vec<f32> = vec![0.0f32; cfg.batch * chunk_frames];
            let mut real_latents: Vec<usize> = vec![0; cfg.batch]; // len thật từng sample

            for i in 0..cfg.batch {
                // Round-robin theo giọng → batch cân bằng bất kể số chunk mỗi giọng.
                let spk = active[cursor % active.len()];
                cursor = (cursor + 1) % active.len();
                let list = &per_spk_chunks[spk];
                let k = list[rng.below(list.len())];
                let chunk = &train_ds.chunks[k];
                micro_utts.push((
                    train_ds.utts[chunk.utt].utt_id.clone(),
                    chunk.offset_lat,
                ));
                let t_chunk = train_ds.latent_len(k).min(chunk_frames);
                real_latents[i] = t_chunk;

                // feats slice phần thật rồi pad 0 về t_h_seg.
                let f = train_ds.chunk_feats_channel_major(k, t_chunk);
                debug_assert!(
                    f.len() <= cfg.batch * c * t_h_seg,
                    "chunk feats vượt chuẩn pad"
                );
                feats_host[i * c * t_h_seg..i * c * t_h_seg + f.len()].copy_from_slice(&f);
                // wav đã tự pad 0 về `n` trong segment_wav.
                let (wav_seg, _) = train_ds.segment_wav(k, 0, chunk_frames);
                wav_host[i * n..(i + 1) * n].copy_from_slice(&wav_seg);

                // f0 của chunk này, resample từ 100 Hz (RMVPE) về frame latent (86.13 Hz)
                if cfg.nsf || cfg.pitch_embed {
                    let f0 = train_ds.utts[chunk.utt].f0.as_ref().expect("f0 đã kiểm ở trên");
                    // f0 offset tương ứng offset_lat của chunk
                    let f0_start = ((chunk.offset_lat as f64 * 100.0 / LATENT_HZ).round() as usize)
                        .min(f0.len().saturating_sub(1));
                    let f0_end = (f0_start + t_chunk).min(f0.len());
                    let f0_slice = &f0[f0_start..f0_end];
                    // nearest_index_map resample f0_slice (100Hz) → chunk_frames (86.13Hz)
                    let map = nearest_index_map(f0_slice.len(), t_chunk);
                    for t in 0..t_chunk {
                        f0_host[i * chunk_frames + t] = f0_slice[map[t]];
                    }
                }
            }

            let b = cfg.batch;
            let feats = Tensor::<3>::from_data(TensorData::new(feats_host, [b, c, t_h_seg]), &device);
            let target = Tensor::<2>::from_data(TensorData::new(wav_host, [b, n]), &device);
            // Pitch bins: quantize f0 (Hz) log-cân đối 256 bin. bin 0 = unvoiced,
            // bin 1..255 = 1 + round(64·log2(f/50)) clamp — voicing 50..~1270 Hz.
            let f0_bins = if cfg.pitch_embed {
                let mut bins_host: Vec<i64> = Vec::with_capacity(b * chunk_frames);
                for i in 0..b {
                    let f0 = &f0_host[i * chunk_frames..(i + 1) * chunk_frames];
                    fill_f0_bins(f0, &mut bins_host);
                }
                Some(
                    Tensor::<2, Int>::from_ints(bins_host.as_slice(), &device)
                        .reshape([b, chunk_frames]),
                )
            } else {
                None
            };
            let f0_tensor = if cfg.nsf {
                Some(Tensor::<2>::from_data(
                    TensorData::new(f0_host, [b, chunk_frames]),
                    &device,
                ))
            } else {
                None
            };

            // Mask frame mel: 1 khi window STFT của frame nằm hoàn toàn trong vùng thật
            // (t·hop + n_fft/2 ≤ real_samples) — loại nhiễu méo từ vùng pad 0.
            let hop = mel_cfg.hop_length as usize;
            let n_fft_half = (mel_cfg.n_fft as usize) / 2;
            let mask_frames = n / hop + 1;
            let mut mask_host = vec![0.0f32; b * mask_frames];
            for i in 0..b {
                let real_samples = real_latents[i] * SAMPLES_PER_FRAME;
                for t in 0..mask_frames {
                    if t * hop + n_fft_half <= real_samples {
                        mask_host[i * mask_frames + t] = 1.0;
                    }
                }
            }

            let z = if cfg.pitch_embed {
                adapter.forward_to_pitch(feats, chunk_frames, f0_bins)
            } else {
                adapter.forward_to(feats, chunk_frames)
            };
            let z_probe = z.clone();
            let wav = gen.forward(z, f0_tensor, false).reshape([b, n]);
            let mask = Tensor::<2>::from_data(TensorData::new(mask_host, [b, mask_frames]), &device);
            let wav_probe = wav.clone();
            let loss = mel::mel_l1_masked(wav, target, &mel_cfg, &basis, &mask);

            // Đọc scalar bằng try_into_scalar (không panic khi kernel lỗi).
            let lv: f64 = loss
                .clone()
                .try_into_scalar::<f32>()
                .map(|v| v as f64)
                .unwrap_or(f64::NAN);
            // Luôn gọi backward() — kể cả khi NaN — để burn giải phóng graph.
            // Không backward ⇒ tensor trung gian lưu trong queue CUDA → VRAM cạn dần.
            let mut grads = GradientsParams::from_grads(loss.backward(), &adapter);
            // Cắt gradient toàn cục nếu L2 norm vượt cfg.clip_norm (chống blowup Adam).
            let gn = clip_grad_l2(&mut grads, &adapter, cfg.clip_norm, &device);
            if gn.is_finite() {
                gn_ema = gn_ema * 0.9 + gn * 0.1;
            }
            if !lv.is_finite() {
                nan_skips += 1;
                // Diagnostics: NaN đến từ đâu? zmax=NaN chỉ ra adapter weights đã
                // bị ô nhiễm; chỉ wavmax=NaN → decoder/f0; cả hai finite → loss/mel.
                let zmax = match z_probe.abs().max().try_into_scalar::<f32>() {
                    Ok(v) => format!("{v:.3}"),
                    Err(_) => "?".to_string(),
                };
                let wavmax = match wav_probe.abs().max().try_into_scalar::<f32>() {
                    Ok(v) => format!("{v:.3}"),
                    Err(_) => "?".to_string(),
                };
                let utts: String = micro_utts
                    .iter()
                    .map(|(u, o)| format!("{u}@{o}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                eprintln!(
                    "step {:>4} micro {}/{} train mel-L1 = NaN → bỏ merge gradient (all-NaN-chuỗi {consec}) gn={gn:.2} zmax={zmax} wavmax={wavmax} utts=[{utts}]",
                    step + 1,
                    _micro + 1,
                    accum,
                );
                continue;
            }
            lv_sum += lv;
            // Cộng dồn gradients micro-batch vào `cum` (theo param id).
            merge_into(&mut cum, &mut grads, &adapter);
        }

        let n_valid = accum - nan_skips;
        let lv = if n_valid > 0 {
            lv_sum / n_valid as f64
        } else {
            f64::NAN
        };
        if step == 0 {
            first = lv;
        }
        last = lv;

        if n_valid > 0 {
            consec = 0; // có ít nhất 1 micro hợp lệ → chuỗi NaN đứt
            let lr_now = lr_at(step, cfg.steps, cfg.warmup, cfg.lr);
            adapter = optim.step(lr_now, adapter, cum);
        } else {
            eprintln!("step {:>4} toàn bộ micro-batch NaN → bỏ optimizer step", step + 1);
            consec += 1;
            if consec >= 3 {
                return Err(format!(
                    "train diverge: {consec} optimizer step liên tục toàn NaN (lr={} quá cao hoặc dữ liệu cần kiểm tra), dừng sớm",
                    cfg.lr
                )
                .into());
            }
        }

        if let Some(path) = &checkpoint_path {
            if (step + 1) % cfg.checkpoint_every == 0 {
                adapter.save(path)?;
                println!("  ↳ checkpoint step {} → {}", step + 1, path.display());
            }
        }

        let do_val =
            cfg.val_every > 0 && ((step + 1) % cfg.val_every == 0 || step + 1 == cfg.steps);
        let val_now = if do_val {
            val_ran = true;
            // Cùng device với train — module tự do Grad tracking khi forward không require_grad.
            let report = eval::evaluate(&adapter, &gen, val_ds, &mel_cfg, &basis, &device, &device)?;
            let improved = report.mean_mel_l1 < best_val;
            if improved {
                best_val = report.mean_mel_l1;
                best_adapter = adapter.clone();
            }
            println!(
                "step {:>4}  train mel-L1 = {:.5}  val = {:.5}{}{}  gn={:.2}{}",
                step + 1,
                lv,
                report.mean_mel_l1,
                if improved { "  ← tốt nhất" } else { "" },
                mem_line(),
                gn_ema,
                if gn_ema > cfg.clip_norm * 0.5 { " ⚠grad-norm cao" } else { "" }
            );
            Some(report.mean_mel_l1)
        } else {
            if step % 25 == 0 || step + 1 == cfg.steps {
                println!("step {:>4}  train mel-L1 = {lv:.5}  gn={gn_ema:.2}{}", step + 1, mem_line());
            }
            None
        };

        history.push((step + 1, lv, val_now));
        write_history(&history_path, &history)?;
    }

    adapter.save(&adapter_path)?;
    // val_every = 0 → không bao giờ val → best_adapter còn là bản khởi tạo random:
    // phải lưu bản train (last) làm best để adapter_best.bpk luôn dùng được.
    if val_ran {
        best_adapter.save(&best_path)?;
    } else {
        adapter.save(&best_path)?;
        eprintln!("⚠️  val_every = 0 — không có val; adapter_best.bpk = bản train cuối (không phải init random)");
    }

    Ok(TrainingSummary {
        first,
        last,
        best_val,
        chunk_frames,
        batch: cfg.batch,
        params,
        adapter_path,
        best_path,
        history_path,
        checkpoint_path,
    })
}

/// Cộng dồn gradients của micro-batch sau vào accumulator (`GradientsParams`
/// Burn 0.22 không có `merge` — cộng tay theo param id: rank 2 (weight) và
/// rank 1 (bias) của `Adapter`).
fn merge_into(cum: &mut GradientsParams, gp: &mut GradientsParams, adapter: &Adapter) {
    for (id, rank) in adapter.param_ids() {
        match rank {
            2 => add_grad::<2>(cum, gp, id),
            1 => add_grad::<1>(cum, gp, id),
            _ => {}
        }
    }
}

fn add_grad<const R: usize>(
    cum: &mut GradientsParams,
    gp: &mut GradientsParams,
    id: burn::module::ParamId,
) {
    match (cum.remove::<R>(id), gp.remove::<R>(id)) {
        (Some(a), Some(b)) => cum.register(id, a + b),
        (None, Some(b)) => cum.register(id, b),
        (Some(a), None) => cum.register(id, a),
        (None, None) => {}
    }
}

/// Cắt gradient toàn cục (global L2 norm): nếu norm > max_norm scale mọi tensor
/// về (max_norm / norm). Trả về norm trước khi cắt (hoặc 0 nếu tắt) để log.
/// NaN norm ⇒ không cắt, gọi biến caller sẽ nhận biết qua `lv.is_finite()`.
fn clip_grad_l2(gp: &mut GradientsParams, adapter: &Adapter, max_norm: f64, device: &Device) -> f64 {
    if max_norm <= 0.0 {
        return 0.0;
    }
    // Tổng bình phương mọi phần tử gradient của mọi tensor học (linear + TCN blocks).
    let mut total_sq: f64 = 0.0;
    for (id, rank) in adapter.param_ids() {
        let v: f64 = match rank {
            2 => gp
                .remove::<2>(id)
                .map(|g| {
                    let v = g
                        .clone()
                        .square()
                        .sum()
                        .try_into_scalar::<f32>()
                        .unwrap_or(f32::NAN) as f64;
                    gp.register(id, g);
                    v
                })
                .unwrap_or(0.0),
            1 => gp
                .remove::<1>(id)
                .map(|g| {
                    let v = g
                        .clone()
                        .square()
                        .sum()
                        .try_into_scalar::<f32>()
                        .unwrap_or(f32::NAN) as f64;
                    gp.register(id, g);
                    v
                })
                .unwrap_or(0.0),
            _ => 0.0,
        };
        total_sq += v;
    }

    let norm = total_sq.sqrt();
    if norm.is_finite() && norm > max_norm {
        let scale = max_norm as f64 / norm;
        for (id, rank) in adapter.param_ids() {
            match rank {
                2 => {
                    if let Some(g) = gp.remove::<2>(id) {
                        let dims = g.dims();
                        let len: usize = dims.iter().product();
                        let scale_t = Tensor::<2>::from_data(
                            TensorData::new(vec![scale as f32; len], dims),
                            device,
                        );
                        gp.register(id, g * scale_t);
                    }
                }
                1 => {
                    if let Some(g) = gp.remove::<1>(id) {
                        let dims = g.dims();
                        let len: usize = dims.iter().product();
                        let scale_t = Tensor::<1>::from_data(
                            TensorData::new(vec![scale as f32; len], dims),
                            device,
                        );
                        gp.register(id, g * scale_t);
                    }
                }
                _ => {}
            }
        }
    }
    norm
}

/// Adam lr với warmup tuyến tính (0 → base trong `warmup` bước) rồi cosine decay
/// xuống 10% `base` tại bước cuối (tinh thần RVC/Applio, tránh blowup hết_epoch).
fn lr_at(step: usize, total: usize, warmup: usize, base: f64) -> f64 {
    if warmup > 0 && step < warmup {
        return base * (step + 1) as f64 / warmup as f64;
    }
    let remaining = total.saturating_sub(warmup);
    if remaining == 0 || step >= total {
        return base * 0.1;
    }
    let progress = (step - warmup) as f64 / remaining as f64;
    base * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()))
}

/// Quantize f0 (Hz) thành 256 bin log-cân đối (Phase 3 pitch-embed):
/// `bin 0 = unvoiced (f0 ≤ 0)`, `bin 1..255 = 1 + round(64·log2(f/50))` clamp.
/// Quãng tám = 64 bin ⇒ độ phân giải ~18.9 cents/bin ở vùng giọng nói;
/// 50 Hz → 1, ~1271 Hz → 254. Ghi đè `out` (tự push đủ `f0.len()` phần tử).
pub(crate) fn fill_f0_bins(f0: &[f32], out: &mut Vec<i64>) {
    for &f in f0 {
        let bin = if f <= 0.0 {
            0
        } else {
            let b = 1.0 + 64.0 * (f as f64 / 50.0).log2();
            (b.round().clamp(1.0, 255.0)) as i64
        };
        out.push(bin);
    }
}

/// RSS + peak (VmHWM) của process tính GB, đọc `/proc/self/status`.
/// Trả None trên nền tảng không-Linux (macOS) → dòng mem bị bỏ qua.
#[cfg(target_os = "linux")]
fn rss_peak_gb() -> Option<(f64, f64)> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut rss = None;
    let mut hwm = None;
    for line in s.lines() {
        let parse = |v: &str| -> Option<f64> {
            v.split_whitespace()
                .next()?
                .parse::<f64>()
                .ok()
                .map(|kb| kb / 1024.0 / 1024.0)
        };
        if let Some(v) = line.strip_prefix("VmRSS:") {
            rss = parse(v);
        } else if let Some(v) = line.strip_prefix("VmHWM:") {
            hwm = parse(v);
        }
    }
    Some((rss?, hwm?))
}

#[cfg(not(target_os = "linux"))]
fn rss_peak_gb() -> Option<(f64, f64)> {
    None
}

/// ` · RSS=x.xGB PEAK=x.xGB` (chỉ Linux; Colab chạy Linux nên luôn có).
fn mem_line() -> String {
    match rss_peak_gb() {
        Some((rss, hwm)) => format!(" · RSS={rss:.1}GB PEAK={hwm:.1}GB"),
        None => String::new(),
    }
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
    use crate::dataset::{Chunk, Dataset, Utterance};
    use crate::eval::per_utt_metrics;
    use crate::mel;
    use voice_lib::vc::Adapter;
    use voice_mel::MelConfig;
    use voice_vits::dec::Generator;
    use std::path::{Path, PathBuf};

    fn make_utt(speaker: usize, utt_id: &str, t_h: usize, latent_frames: usize) -> Utterance {
        Utterance {
            speaker,
            utt_id: utt_id.to_string(),
            t_h,
            c: 768,
            feats: vec![0.0f32; t_h * 768],
            wav: vec![0.1f32; latent_frames * SAMPLES_PER_FRAME],
            f0: None,
        }
    }

    fn make_ds(chunk_frames: usize) -> Dataset {
        // 1 câu ngắn (1 frame latent) sẽ bị bỏ khi chunk_frames > 1
        // 5 câu dài (40 frame HuBERT ≈ 69 latent) → mỗi câu 1 chunk nếu chunk_frames <= 69
        let mut utts = vec![make_utt(0, "short", 1, 1)];
        for i in 0..5usize {
            utts.push(make_utt(0, format!("long{i}").as_str(), 40, 69));
        }
        let mut per_speaker = vec![Vec::new(); 1];
        for (i, u) in utts.iter().enumerate() {
            per_speaker[u.speaker].push(i);
        }
        // Build chunks for given chunk_frames (0 không hợp lệ — Tests luôn truyền ≥ MIN_SEG)
        let mut chunks = Vec::new();
        let mut per_speaker_chunks = vec![Vec::new(); 1];
        for (u_idx, utt) in utts.iter().enumerate() {
            let total = Dataset::latent_len_of_utt(utt);
            if total >= chunk_frames {
                // 1 chunk per utterance (since total < 2*chunk_frames)
                let chunk = Chunk {
                    utt: u_idx,
                    offset_lat: 0,
                    len_lat: chunk_frames,
                    chunk_id: format!("{}#0", utt.utt_id),
                };
                let cidx = chunks.len();
                chunks.push(chunk);
                per_speaker_chunks[0].push(cidx);
            }
        }
        Dataset {
            speakers: vec!["spk0".to_string()],
            utts,
            per_speaker,
            chunks,
            per_speaker_chunks,
        }
    }

    #[test]
    fn chunk_frames_zero_is_rejected() {
        if !Path::new("sources/vits_ljs_dec/config.json").exists() {
            eprintln!("skip: sources/vits_ljs_dec thiếu");
            return;
        }
        let device = Device::flex();
        let train_ds = make_ds(64);
        let val_ds = make_ds(64);

        let cfg = TrainConfig {
            dec_dir: PathBuf::from("sources/vits_ljs_dec"),
            out_dir: std::env::temp_dir().join("voice_rs_chunk_test_train"),
            chunk_frames: 0,
            batch: 2,
            grad_accum: 1,
            steps: 5,
            lr: 1e-4,
            val_every: 5,
            seed: 42,
            warmup: 100,
            clip_norm: 1.0,
            checkpoint_every: 0,
            nsf: false,
            pitch_embed: false,
        };
        let err = run_training(&train_ds, &val_ds, &cfg, &device).unwrap_err();
        assert!(
            err.to_string().contains("không còn hỗ trợ"),
            "lỗi phải nêu rõ fallback bị bỏ: {err}"
        );
    }

    #[test]
    fn eval_chunks_work() {
        if !Path::new("sources/vits_ljs_dec/config.json").exists() {
            eprintln!("skip: sources/vits_ljs_dec thiếu");
            return;
        }
        let device = Device::flex();
        let ds = make_ds(64); // chunk_frames=64

        let gen = Generator::load(&PathBuf::from("sources/vits_ljs_dec"), &device)
            .expect("nạp dec")
            .freeze();
        let mel_cfg = MelConfig::ljs();
        let basis = mel::mel_basis(&mel_cfg, &device);
        let adapter = Adapter::new(&device).unfreeze();

        let metrics = per_utt_metrics(&adapter, &gen, &ds, &mel_cfg, &basis, &device, &device)
            .expect("per_utt_metrics không panic");
        // 5 câu dài → 5 chunks (câu ngắn bị bỏ)
        assert_eq!(metrics.len(), 5, "chỉ chunk dài được giữ");
        assert!(metrics.iter().all(|m| m.mel_l1.is_finite()));
    }

    #[test]
    fn fill_f0_bins_quantizes_log_scale() {
        let mut bins = Vec::new();
        fill_f0_bins(&[0.0, -3.0, 50.0, 100.0, 200.0, 400.0], &mut bins);
        // unvoiced (≤0) → bin 0
        assert_eq!(bins[0], 0);
        assert_eq!(bins[1], 0);
        // mỗi nhân 2 của f0 = +64 bin: 50→1, 100→65, 200→129, 400→193
        assert_eq!(bins[2], 1);
        assert_eq!(bins[3], 65);
        assert_eq!(bins[4], 129);
        assert_eq!(bins[5], 193);

        // clamp biên: cả hai đầu
        let mut b2 = Vec::new();
        fill_f0_bins(&[1.0, 5000.0], &mut b2);
        assert_eq!(b2[0], 1, "f0 rất thấp clamp về bin 1");
        assert_eq!(b2[1], 255, "f0 rất cao clamp về bin 255");
    }

    #[test]
    fn lr_at_warmup_then_cosine() {
        let base = 1e-4;
        // warmup: step 0 → base/warmup (1e-6), cuối warmup → base
        assert!((lr_at(0, 1000, 100, base) - base / 100.0).abs() < 1e-12);
        assert!((lr_at(99, 1000, 100, base) - base).abs() < 1e-12);
        // bước đầu sau warmup vẫn ≈ base (cos(0) = 1)
        assert!((lr_at(100, 1000, 100, base) - base).abs() < 1e-12);
        // giữa đường cos nằm trong (0.1·base, base)
        let mid = lr_at(500, 1000, 100, base);
        assert!(mid > base * 0.1 && mid < base);
        // cuối = 0.1·base
        assert!((lr_at(999, 1000, 100, base) - base * 0.1).abs() < 1e-9);
        // warmup = 0 → luôn base (trừ bước cuối epoch)
        assert!((lr_at(0, 100, 0, base) - base).abs() < 1e-12);
    }

    #[test]
    fn clip_grad_l2_scales_and_returns_norm() {
        let device = Device::flex();
        let adapter = Adapter::new(&device);
        // param_ids[0] = linear.weight (rank 2), [1] = linear.bias (rank 1)
        let w_id = adapter.param_ids()[0].0;
        let b_id = adapter.param_ids()[1].0;
        let mut gp = GradientsParams::new();
        // weight [[3,4],[0,0]] → norm² 25; bias [1,0] → norm² 1; tổng norm = sqrt(26)
        gp.register(
            w_id,
            Tensor::<2>::from_data(
                TensorData::new(vec![3.0f32, 4.0, 0.0, 0.0], [2, 2]),
                &device,
            ),
        );
        gp.register(
            b_id,
            Tensor::<1>::from_data(TensorData::new(vec![1.0f32, 0.0], [2]), &device),
        );
        let norm = clip_grad_l2(&mut gp, &adapter, 1.0, &device);
        assert!((norm - (26.0f64).sqrt()).abs() < 1e-5, "norm = {norm}");

        // sau clip: mỗi tensor chia norm → tổng norm mới = 1.0
        let w = gp.remove::<2>(w_id).unwrap();
        let wv = w.into_data().as_slice::<f32>().unwrap().to_vec();
        let n = (26.0f64).sqrt() as f32;
        assert!((wv[0] - 3.0 / n).abs() < 1e-4, "w0 = {}", wv[0]);
        assert!((wv[1] - 4.0 / n).abs() < 1e-4, "w1 = {}", wv[1]);
        let b = gp.remove::<1>(b_id).unwrap();
        let bv = b.into_data().as_slice::<f32>().unwrap().to_vec();
        assert!((bv[0] - 1.0 / n).abs() < 1e-4, "b0 = {}", bv[0]);
    }

    #[test]
    fn clip_grad_l2_disabled_when_max_norm_zero() {
        let device = Device::flex();
        let adapter = Adapter::new(&device);
        let w_id = adapter.param_ids()[0].0;
        let mut gp = GradientsParams::new();
        gp.register(
            w_id,
            Tensor::<2>::from_data(
                TensorData::new(vec![10.0f32, 0.0, 0.0, 0.0], [2, 2]),
                &device,
            ),
        );
        let norm = clip_grad_l2(&mut gp, &adapter, 0.0, &device);
        assert_eq!(norm, 0.0, "clip tắt phải trả 0");
        // tensor không bị thay đổi
        let w = gp.remove::<2>(w_id).unwrap();
        let wv = w.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(wv, vec![10.0f32, 0.0, 0.0, 0.0]);
    }
}