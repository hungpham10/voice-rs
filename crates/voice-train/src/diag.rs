//! Chẩn đoán **rè/nhiễu** — mel-L1 theo band tần số + voiced/unvoiced (PR-B1).
//!
//! Mục đích: phân biệt hai nguyên nhân rè trước khi xây NSF (§13.4.1):
//! - **(a) z ngoài manifold** (bridge lỗi): mọi band đều lệch, kể cả unvoiced;
//! - **(b) thiếu excitation**: lỗi tập trung ở **voiced** frames (dec không biết
//!   chu kỳ nào để tái tạo) — 독 mid/high, unvoiced thường ổn hơn rõ rệt.
//!
//! Tất định: cùng 2 wav vào → cùng bảng số (drift = 0).
//!
//! ```bash
//! voice-train diag --target target.wav --gen gen.wav [--f0 f0.npy] [--out-csv diag.csv]
//! ```

use std::path::Path;

use burn::tensor::{Device, Tensor, TensorData};
use voice_lib::vc::{nearest_index_map, resample_linear};
use voice_mel::MelConfig;

/// Các band tần số phân tích (Hz).
pub const BANDS: &[(f64, f64, &str)] = &[
    (80.0, 1_000.0, "low80-1k"),
    (1_000.0, 4_000.0, "mid1k-4k"),
    (4_000.0, 8_000.0, "high4k-8k"),
];

/// Kết quả chẩn đoán một cặp (target, gen).
#[derive(Debug, Clone)]
pub struct DiagSummary {
    /// mel-L1 toàn phổ.
    pub overall: f64,
    /// mel-L1 trung bình mỗi band (cùng thứ tự [`BANDS`]).
    pub bands: Vec<(String, f64)>,
    /// mel-L1 trên **voiced** frames (f0 > 0) — nếu có f0.
    pub voiced: Option<f64>,
    /// mel-L1 trên **unvoiced** frames (f0 == 0) — nếu có f0.
    pub unvoiced: Option<f64>,
    /// % frame voiced. `None` nếu không có f0.
    pub voiced_pct: Option<f64>,
    /// Số frame đã so.
    pub frames: usize,
}

impl DiagSummary {
    /// In bảng tóm tắt ra stdout.
    pub fn print(&self) {
        println!("== diag ({} frame) ==", self.frames);
        println!("mel-L1 toàn phổ : {:.5}", self.overall);
        for (name, v) in &self.bands {
            println!("  {name:<12} : {v:.5}");
        }
        if let Some(pct) = self.voiced_pct {
            println!(
                "  voiced      : {:.5}  ({} % frame)",
                self.voiced.unwrap_or(f64::NAN),
                pct
            );
            println!("  unvoiced    : {:.5}", self.unvoiced.unwrap_or(f64::NAN));
        }
    }
}

/// Tần số trung tâm (Hz) của mel-bin thứ `m` trong `n_mels` bin.
///
/// Bin nằm đều trên trục mel: `center(m) = mel_to_hz(mel(fmin) + (m+0.5)·Δmel)`.
fn mel_bin_hz(cfg: &MelConfig, m: usize) -> f64 {
    let mel_min = rosa::stft::hz_to_mel(cfg.fmin as f64, cfg.htk);
    let mel_max = rosa::stft::hz_to_mel(cfg.fmax as f64, cfg.htk);
    let step = (mel_max - mel_min) / cfg.n_mels as f64;
    rosa::stft::mel_to_hz(mel_min + (m as f64 + 0.5) * step, cfg.htk)
}

/// Chỉ số mel-bin thuộc band `(lo, hi)`.
fn bins_of_band(cfg: &MelConfig, lo: f64, hi: f64) -> Vec<usize> {
    (0..cfg.n_mels)
        .filter(|&m| {
            let f = mel_bin_hz(cfg, m);
            lo <= f && f < hi
        })
        .collect()
}

/// Đılması `f0.npy` (RMVPE @ 100 Hz): shape `[1, T]` hoặc `[1, 1, T]`.
fn read_f0(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (shape, flat) = voice_vits::npy::read_f32(path)?;
    match shape.as_slice() {
        [1, _t] | [1, 1, _t] => Ok(flat),
        other => Err(format!("f0 phải là [1,T]/[1,1,T], được {other:?}").into()),
    }
}

/// Căn target/gen về cùng `n` mẫu @ 22.05 kHz, trả cả hai.
fn align_pair(
    target_wav: &Path,
    gen_wav: &Path,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let (mut tgt, sr) = crate::data::read_wav_mono(target_wav)?;
    let (mut gen, sr_gen) = crate::data::read_wav_mono(gen_wav)?;
    if sr != 22050 {
        tgt = resample_linear(&tgt, sr, 22050);
    }
    if sr_gen != 22050 {
        gen = resample_linear(&gen, sr_gen, 22050);
    }
    let n = tgt.len().min(gen.len());
    if n < 1024 {
        return Err(format!("wav quá ngắn để chẩn đoán: {n} mẫu").into());
    }
    tgt.truncate(n);
    gen.truncate(n);
    Ok((tgt, gen))
}

/// Chạy chẩn đoán. Trả `(tóm tắt, bảng từng frame)`.
///
/// Bảng `rows` mỗi dòng `Vec<f64>`: `[overall, low, mid, high, voiced_flag]`
/// (voiced_flag = 1/0 hoặc NaN khi không có f0) — dùng cho `--out-csv`.
pub fn diagnose(
    target_wav: &Path,
    gen_wav: &Path,
    f0_path: Option<&Path>,
    device: &Device,
) -> Result<(DiagSummary, Vec<Vec<f64>>), Box<dyn std::error::Error>> {
    let (tgt, gen) = align_pair(target_wav, gen_wav)?;
    let n = tgt.len();

    let cfg = MelConfig::ljs();
    let basis = crate::mel::mel_basis(&cfg, device); // [n_freq, n_mels]
    let to_t = |wav: Vec<f32>| Tensor::<2>::from_data(TensorData::new(wav, [1, n]), device);
    let a = crate::mel::log_mel(to_t(tgt), &cfg, &basis); // [1, F, 80]
    let b = crate::mel::log_mel(to_t(gen), &cfg, &basis);
    let data = (a - b).abs().into_data().convert::<f64>();

    let [_, frames, n_mels] = data.shape().as_ref() else {
        return Err("shape mel diff phải là [1, F, 80]".into());
    };
    let (frames, n_mels) = (*frames, *n_mels);
    let flat = data
        .as_slice::<f64>()
        .map_err(|e| format!("lỗi đọc diff: {e}"))?
        .to_vec();
    // flat[t * n_mels + m]
    let cell = |t: usize, m: usize| flat[t * n_mels + m];

    // ---- mean mỗi frame [t]
    let per_frame: Vec<f64> = (0..frames)
        .map(|t| (0..n_mels).map(|m| cell(t, m)).sum::<f64>() / n_mels as f64)
        .collect();

    // ---- mean mỗi band
    let band_bins: Vec<Vec<usize>> = BANDS
        .iter()
        .map(|(lo, hi, _)| bins_of_band(&cfg, *lo, *hi))
        .collect();
    let bands = BANDS
        .iter()
        .zip(&band_bins)
        .map(|((_, _, name), bins)| {
            let v = if bins.is_empty() {
                f64::NAN
            } else {
                bins.iter()
                    .map(|&m| (0..frames).map(|t| cell(t, m)).sum::<f64>())
                    .sum::<f64>()
                    / ((bins.len() * frames) as f64)
            };
            (name.to_string(), v)
        })
        .collect();

    let overall = flat.iter().sum::<f64>() / flat.len() as f64;

    // ---- voiced / unvoiced từ f0 @ 100 Hz (nearest map về frame rate mel)
    let (voiced, unvoiced, voiced_pct, voiced_flags) = match f0_path {
        Some(p) => {
            let f0 = read_f0(p)?;
            let idx = nearest_index_map(f0.len(), frames);
            let flags: Vec<bool> = idx.iter().map(|&i| f0[i] > 0.0).collect();
            let c_v = flags.iter().filter(|&&b| b).count();
            let mean_of = |want: bool| -> Option<f64> {
                let (sum, cnt) = per_frame
                    .iter()
                    .zip(&flags)
                    .fold((0.0, 0usize), |(s, c), (&v, &b)| {
                        if b == want {
                            (s + v, c + 1)
                        } else {
                            (s, c)
                        }
                    });
                (cnt > 0).then(|| sum / cnt as f64)
            };
            (
                mean_of(true),
                mean_of(false),
                Some(100.0 * c_v as f64 / frames.max(1) as f64),
                flags,
            )
        }
        None => (None, None, None, vec![false; frames]),
    };

    let rows: Vec<Vec<f64>> = (0..frames)
        .map(|t| {
            let mut row = vec![per_frame[t]];
            for (i, _) in BANDS.iter().enumerate() {
                // mean band của riêng frame t
                let bins = &band_bins[i];
                let v = if bins.is_empty() {
                    f64::NAN
                } else {
                    bins.iter().map(|&m| cell(t, m)).sum::<f64>() / bins.len() as f64
                };
                row.push(v);
            }
            if f0_path.is_some() {
                row.push(if voiced_flags[t] { 1.0 } else { 0.0 });
            }
            row
        })
        .collect();

    Ok((
        DiagSummary {
            overall,
            bands,
            voiced,
            unvoiced,
            voiced_pct,
            frames,
        },
        rows,
    ))
}

/// Ghi CSV từng frame (kết quả [`diagnose`]`.1`).
pub fn write_csv(
    path: &Path,
    rows: &[Vec<f64>],
    f0_present: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut s = String::from("frame,overall,low80_1k,mid1k_4k,high4k_8k");
    if f0_present {
        s.push_str(",voiced");
    }
    s.push('\n');
    for (t, row) in rows.iter().enumerate() {
        s.push_str(&t.to_string());
        for v in row.iter() {
            s.push_str(&format!(",{v:.5}"));
        }
        s.push('\n');
    }
    std::fs::write(path, s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(n: usize, sr: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * core::f32::consts::PI * 220.0 * (i as f32 / sr)).sin() * 0.5)
            .collect()
    }

    fn write_wav(path: &Path, samples: &[f32], sr: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for &s in samples {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
    }

    #[test]
    fn identical_audio_zero_everywhere() {
        let device = Device::flex();
        let dir = std::env::temp_dir().join("voice_rs_diag_test");
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.wav");
        let x = sine(3 * 22050, 22050.0);
        write_wav(&a, &x, 22050);
        let (sum, _) = diagnose(&a, &a, None, &device).unwrap();
        assert!(sum.overall < 1e-4, "overall phải ~0, got {}", sum.overall);
        for (_, v) in &sum.bands {
            assert!(*v < 1e-4, "band phải ~0, got {v}");
        }
        assert!(sum.voiced.is_none());
        assert_eq!(sum.frames, 1 + 3 * 22050 / 256);
    }

    #[test]
    fn diff_audio_positive_and_bands_complete() {
        let device = Device::flex();
        let dir = std::env::temp_dir().join("voice_rs_diag_test2");
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.wav");
        let b = dir.join("b.wav");
        // 220 Hz vs 440 Hz — khác biệt rõ ràng mọi band
        write_wav(&a, &sine(2 * 22050, 22050.0), 22050);
        write_wav(&b, &sine_ln(440.0), 22050);
        let (sum, rows) = diagnose(&a, &b, None, &device).unwrap();
        assert!(sum.overall > 0.01, "phải thấy khác biệt: {}", sum.overall);
        assert_eq!(sum.bands.len(), BANDS.len());
        for (_, v) in &sum.bands {
            assert!(*v > 0.0 && v.is_finite(), "band phải finite >0, got {v}");
        }
        assert_eq!(rows.len(), sum.frames);
        assert_eq!(rows[0].len(), BANDS.len() + 1); // không có f0 → không cột voiced
    }

    fn sine_ln(f: f32) -> Vec<f32> {
        (0..2 * 22050)
            .map(|i| (2.0 * core::f32::consts::PI * f * (i as f32 / 22050.0)).sin() * 0.5)
            .collect()
    }

    #[test]
    fn f0_mask_splits_voiced_unvoiced() {
        let device = Device::flex();
        let dir = std::env::temp_dir().join("voice_rs_diag_test3");
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.wav");
        let b = dir.join("b.wav");
        write_wav(&a, &sine(2 * 22050, 22050.0), 22050);
        write_wav(&b, &sine_ln(660.0), 22050);

        // f0: nửa đầu voiced (220 Hz), nửa sau unvoiced (0) — @100 Hz
        let frames_f0 = 2 * 22050 / 160;
        let mut f0 = vec![220.0f32; frames_f0 / 2];
        f0.resize(frames_f0, 0.0);
        let p = dir.join("f0.npy");
        write_npy_f32(&p, &[1, frames_f0], &f0).expect("ghi f0.npy");

        let (sum, _) = diagnose(&a, &b, Some(&p), &device)
            .map_err(|e| format!("diag lỗi: {e}"))
            .unwrap();
        assert!(sum.voiced.is_some(), "phải tách voiced/unvoiced");
        assert_eq!(sum.voiced.is_some(), sum.unvoiced.is_some());
        assert!(sum.voiced_pct.unwrap() > 30.0 && sum.voiced_pct.unwrap() < 70.0);
    }

    /// Ghi một file `.npy` f32 (shape đúng như truyền vào, ví dụ `[1, T]`).
    fn write_npy_f32(path: &Path, shape: &[usize], flat: &[f32]) -> Result<(), std::io::Error> {
        use std::io::Write;
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let mut common = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims},), }}"
        );
        // padding để tổng header chia hết cho 64, kết thúc bằng "\n"
        while common.len() < 60 {
            common.push(' ');
        }
        common.push('\n');
        let mut f = std::fs::File::create(path)?;
        f.write_all(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0])?;
        let hlen = common.len() as u16;
        f.write_all(&hlen.to_le_bytes())?;
        f.write_all(common.as_bytes())?;
        f.write_all(
            &flat.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        Ok(())
    }
}
