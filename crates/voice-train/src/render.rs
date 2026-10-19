//! `voice-train render` — **suy diễn thuần**: nạp features + adapter đã train →
//! sinh wav (mặc định **cả câu**) qua `dec` đóng băng. Không huấn luyện, không
//! đụng tới trọng số — dùng để nghe lại bất kỳ adapter nào, bất kỳ lúc nào.
//!
//! ```text
//! feats [1,768,T_h] @ 50 Hz ── Adapter (đã train) ── z [1,192,t_out] ── dec ── wav 22.05 kHz
//! ```

use std::path::PathBuf;

use burn::tensor::{Device, Tensor};
use voice_lib::vc::Adapter;
use voice_vits::dec::Generator;

use crate::data;
use crate::dataset::{HUBERT_HZ, LATENT_HZ, SAMPLES_PER_FRAME};
use crate::streaming::{chunk_ctx_span, energy_profile_feats, plan_chunks, PEAK, SIL_RMS};

/// Sample rate đầu ra của `dec`.
pub const OUT_SR: u32 = 22050;

/// Cấu hình suy diễn.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub dec_dir: PathBuf,
    pub feats: PathBuf,
    pub adapter: PathBuf,
    pub out_wav: PathBuf,
    /// `0` = cả câu; `>0` = giới hạn tổng frame latent (alias `max_frames`).
    pub max_frames: usize,
    /// Số frame latent mỗi chunk (0 = cả câu; mặc định 512).
    pub chunk_frames: usize,
    /// Context frames ± mỗi bên cho gen (mặc định 32).
    pub ctx_frames: usize,
    /// Bỏ qua chunk im lặng (RMS < SIL_RMS) — copy zeros thay vì gen.
    pub skip_silence: bool,
}

/// Kết quả suy diễn.
#[derive(Debug, Clone)]
pub struct RenderSummary {
    pub t_out: usize,
    pub samples: usize,
    pub seconds: f64,
    pub out_wav: PathBuf,
}

/// Số frame latent (`86.13 Hz`) suy được từ `t_h` frame HuBERT (`50 Hz`) — làm tròn xuống.
pub fn latent_len_for(t_h: usize) -> usize {
    (t_h as f64 * LATENT_HZ / HUBERT_HZ).floor() as usize
}

/// Sinh wav từ features + adapter. `device` không cần autodiff (chỉ forward).
pub fn render(
    cfg: &RenderConfig,
    device: &Device,
) -> Result<RenderSummary, Box<dyn std::error::Error>> {
    if !cfg.dec_dir.join("config.json").exists() {
        return Err(format!(
            "{} chưa có — chạy ./scripts/fetch_assets.sh",
            cfg.dec_dir.display()
        )
        .into());
    }
    if !cfg.feats.exists() {
        return Err(format!(
            "{} chưa có — chạy `embed <wav> <out.npy>` trước",
            cfg.feats.display()
        )
        .into());
    }
    if !cfg.adapter.exists() {
        return Err(format!("{} chưa có — chạy `train …` trước", cfg.adapter.display()).into());
    }

    let gen = Generator::load(&cfg.dec_dir, device)?;
    let adapter = Adapter::load(&cfg.adapter, device)?;

    let feats_full = data::features_channel_major(&cfg.feats, device)?; // [1,768,T_h]
    let t_h_full = feats_full.dims()[2];
    if t_h_full == 0 {
        return Err(format!("{} rỗng", cfg.feats.display()).into());
    }

    let t_out_full = latent_len_for(t_h_full);
    let t_out_total = if cfg.max_frames == 0 {
        t_out_full
    } else {
        cfg.max_frames.min(t_out_full)
    }
    .max(1);

    // Backward compat: chunk_frames == 0 → render nguyên câu (cũ)
    if cfg.chunk_frames == 0 {
        let t_h = ((t_out_total as f64 * HUBERT_HZ / LATENT_HZ).round() as usize)
            .min(t_h_full)
            .max(1);
        let feats = feats_full.narrow(2, 0, t_h);

        let z = adapter.forward_to(feats, t_out_total);
        let samples = t_out_total * SAMPLES_PER_FRAME;
        let wav = gen
            .forward(z, None, false)
            .reshape([1, samples])
            .into_data()
            .as_slice::<f32>()?
            .to_vec();
        data::write_wav_mono(&cfg.out_wav, &wav, OUT_SR)?;

        return Ok(RenderSummary {
            t_out: t_out_total,
            samples,
            seconds: samples as f64 / OUT_SR as f64,
            out_wav: cfg.out_wav.clone(),
        });
    }

    // ===== Chunking kiểu Applio: split → gen với context → trim → concat =====
    let chunk_frames = cfg.chunk_frames;
    let ctx = cfg.ctx_frames;

    // Energy profile từ features (L2 norm theo frame latent) để tìm điểm cắt
    let feats_full_host = feats_full.into_data();
    let feats_full_vec = feats_full_host.as_slice::<f32>()?.to_vec();
    let energy = energy_profile_feats(&feats_full_vec, t_h_full, 768);

    // Plan chunks: chỉ xét t_out_total frame
    let chunks = plan_chunks(t_out_total, chunk_frames, &energy);
    if chunks.is_empty() {
        return Err("không có chunk hợp lệ để render (câu quá ngắn?)".into());
    }

    println!(
        "render {} chunks (chunk={}, ctx={}) tổng {} frame",
        chunks.len(),
        chunk_frames,
        ctx,
        t_out_total
    );

    let mut out_wav = Vec::new();
    let mut total_rendered = 0usize;

    for (idx, chunk) in chunks.iter().enumerate() {
        let (feat_start, feat_len, trim_left) = chunk_ctx_span(*chunk, t_out_total, ctx);

        // Tính t_h tương ứng cho feat_len
        let t_h_seg = ((feat_len as f64 * HUBERT_HZ / LATENT_HZ).round() as usize).max(1);
        let t_h_start = ((feat_start as f64 * HUBERT_HZ / LATENT_HZ).floor() as usize).min(t_h_full.saturating_sub(1));

        // Slice features [feat_start, feat_start + feat_len) → channel-major
        let c = 768;
        let mut feats_seg = vec![0.0f32; c * t_h_seg];
        let actual_t_h = (feat_start + feat_len).min(t_out_total);
        // Convert latent range to hubert range
        let h0 = t_h_start;
        let h1 = ((actual_t_h as f64 * HUBERT_HZ / LATENT_HZ).ceil() as usize).min(t_h_full);
        let t_h_actual = h1 - h0;
        for ch in 0..c {
            for t in 0..t_h_actual {
                feats_seg[ch * t_h_seg + t] = feats_full_vec[ch * t_h_full + h0 + t];
            }
        }
        // Pad nếu thiếu
        if t_h_actual < t_h_seg {
            for ch in 0..c {
                for t in t_h_actual..t_h_seg {
                    feats_seg[ch * t_h_seg + t] = 0.0;
                }
            }
        }

        let feats_tensor = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(feats_seg, [1, c, t_h_seg]),
            device,
        );
        let z = adapter.forward_to(feats_tensor, feat_len);
        let samples = feat_len * SAMPLES_PER_FRAME;
        let wav = gen
            .forward(z, None, false)
            .reshape([1, samples])
            .into_data()
            .as_slice::<f32>()?
            .to_vec();

        // Trim context: bỏ trim_left frame đầu (×256 samples)
        let trim_samples = trim_left * SAMPLES_PER_FRAME;
        let core_wav = if trim_samples < wav.len() {
            &wav[trim_samples..]
        } else {
            &wav[..]
        };

        // Skip silence: nếu chunk này im lặng (energy thấp) và bật skip_silence
        if cfg.skip_silence && idx < energy.len() {
            let chunk_energy = &energy[chunk.0..(chunk.0 + chunk.1).min(energy.len())];
            let mean_rms = chunk_energy.iter().sum::<f32>() / chunk_energy.len() as f32;
            if mean_rms < SIL_RMS {
                // Push zeros thay vì gen
                out_wav.extend(std::iter::repeat(0.0).take(core_wav.len()));
                continue;
            }
        }

        out_wav.extend_from_slice(core_wav);
        total_rendered += core_wav.len() / SAMPLES_PER_FRAME;
    }

    // Peak normalize 0.99 (như Applio)
    let peak = out_wav.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 && peak != PEAK {
        let scale = PEAK / peak;
        for v in &mut out_wav {
            *v *= scale;
        }
    }

    data::write_wav_mono(&cfg.out_wav, &out_wav, OUT_SR)?;

    Ok(RenderSummary {
        t_out: total_rendered,
        samples: out_wav.len(),
        seconds: out_wav.len() as f64 / OUT_SR as f64,
        out_wav: cfg.out_wav.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_len_matches_husbert_ratio() {
        // 342 frame HuBERT (6.84 s @50 Hz) → 589 frame latent ≈ 6.84 s @86.13 Hz.
        assert_eq!(
            latent_len_for(342),
            (342.0 * LATENT_HZ / HUBERT_HZ).floor() as usize
        );
        assert_eq!(latent_len_for(342), 589);
        assert_eq!(latent_len_for(0), 0);
    }
}
