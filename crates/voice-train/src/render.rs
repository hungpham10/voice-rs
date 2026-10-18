//! `voice-train render` — **suy diễn thuần**: nạp features + adapter đã train →
//! sinh wav (mặc định **cả câu**) qua `dec` đóng băng. Không huấn luyện, không
//! đụng tới trọng số — dùng để nghe lại bất kỳ adapter nào, bất kỳ lúc nào.
//!
//! ```text
//! feats [1,768,T_h] @ 50 Hz ── Adapter (đã train) ── z [1,192,t_out] ── dec ── wav 22.05 kHz
//! ```

use std::path::PathBuf;

use burn::tensor::Device;
use voice_lib::vc::Adapter;
use voice_models::dec::Generator;

use crate::data;
use crate::dataset::{HUBERT_HZ, LATENT_HZ, SAMPLES_PER_FRAME};

/// Sample rate đầu ra của `dec`.
pub const OUT_SR: u32 = 22050;

/// Cấu hình suy diễn.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub dec_dir: PathBuf,
    pub feats: PathBuf,
    pub adapter: PathBuf,
    pub out_wav: PathBuf,
    /// `0` = cả câu; `>0` = giới hạn số frame latent.
    pub seg: usize,
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

    // `seg` = 0 → lấy hết mọi frame latent suy được từ features.
    let t_out_full = latent_len_for(t_h_full);
    let t_out = if cfg.seg == 0 {
        t_out_full
    } else {
        cfg.seg.min(t_out_full)
    }
    .max(1);
    let t_h = ((t_out as f64 * HUBERT_HZ / LATENT_HZ).round() as usize)
        .min(t_h_full)
        .max(1);
    let feats = feats_full.narrow(2, 0, t_h);

    let z = adapter.forward_to(feats, t_out); // [1,192,t_out]
    let samples = t_out * SAMPLES_PER_FRAME;
    let wav = gen
        .forward(z, None, false)
        .reshape([1, samples])
        .into_data()
        .as_slice::<f32>()?
        .to_vec();
    data::write_wav_mono(&cfg.out_wav, &wav, OUT_SR)?;

    Ok(RenderSummary {
        t_out,
        samples,
        seconds: samples as f64 / OUT_SR as f64,
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
