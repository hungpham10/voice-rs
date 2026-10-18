//! `eval` — đo chất lượng adapter đã huấn luyện và ra **cổng PASS/FAIL** (§13.6.4g).
//!
//! Các chỉ số (tất cả tính **xác định** để chạy lại cho cùng kết quả):
//!
//! - `mel_l1` — L1 log-mel trung bình giữa audio sinh và audio đích (từng câu).
//! - `speaker_gap` — độ chênh tương đối giữa giọng tốt nhất và kém nhất:
//!   `(max − min) / mean`. Nhỏ = adapter không thiên vị một giọng.
//! - `rms` — biên độ audio sinh (bắt lỗi im lặng / nổ).
//! - `drift` — chênh lệch giữa hai lần chạy eval (do người gọi tính). Phải ≈ 0.
//!
//! Speaker-similarity (giống giọng đích) **chưa** đo được ở Luồng 1 — cần
//! embedding ContentVec so sánh và chỉ có ý nghĩa sau Luồng 2 (timbre). Ghi rõ
//! `N/A` trong báo cáo thay vì bịa con số.

use std::collections::BTreeMap;
use std::path::Path;

use burn::tensor::{Device, Tensor, TensorData};
use voice_lib::vc::Adapter;
use voice_mel::MelConfig;
use voice_vits::dec::Generator;

use crate::dataset::Dataset;
use crate::mel;

/// Đáy đoạn cần có để tính mel-l1 (khớp MIN_SEG trong train.rs): STFT cần >
/// n_fft/2 = 512 mẫu → ≥ 3 frame; lấy 32 frame để loss ổn định.
const MIN_SEG: usize = 32;

/// Chỉ số của một câu.
#[derive(Debug, Clone)]
pub struct UttMetric {
    pub utt_id: String,
    pub speaker_id: String,
    pub mel_l1: f64,
    pub rms: f64,
    pub samples: usize,
    pub finite: bool,
}

/// Thống kê theo giọng.
#[derive(Debug, Clone)]
pub struct SpeakerStat {
    pub count: usize,
    pub mean_mel_l1: f64,
}

/// Toàn bộ báo cáo eval.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub per_utt: Vec<UttMetric>,
    pub mean_mel_l1: f64,
    pub per_speaker: BTreeMap<String, SpeakerStat>,
    /// `(max − min) / mean` trên mel-L1 trung bình từng giọng.
    pub speaker_gap: f64,
    pub min_rms: f64,
    pub max_rms: f64,
    pub all_finite: bool,
    /// Chênh lệch mel-L1 giữa hai lần chạy (0 = tất định). Do người gọi điền.
    pub drift: f64,
}

/// Tính chỉ số từng câu (một lượt sinh). Dùng lại cho cả `evaluate` và `train`.
pub fn per_utt_metrics(
    adapter: &Adapter,
    gen_eval: &Generator,
    ds: &Dataset,
    cfg: &MelConfig,
    basis: &Tensor<2>,
    device: &Device,
    device_eval: &Device,
) -> Result<Vec<UttMetric>, Box<dyn std::error::Error>> {
    let c = ds.check_dims()?;
    let mut out = Vec::with_capacity(ds.utts.len());
    let mut skipped = 0usize;

    for (u, utt) in ds.utts.iter().enumerate() {
        let t_out = ds.latent_len(u);
        if t_out < MIN_SEG {
            skipped += 1;
            continue;
        }
        let t_out = t_out.max(1);
        let t_h = Dataset::hubert_frames_for(t_out).min(utt.t_h).max(1);
        let host = ds.full_feats_channel_major(u, t_out);
        debug_assert_eq!(host.len(), c * t_h);

        let feats = Tensor::<3>::from_data(TensorData::new(host, [1, c, t_h]), device);
        let z = adapter.forward_to(feats, t_out);
        let n = t_out * crate::dataset::SAMPLES_PER_FRAME;
        let z_eval = z.to_device(device_eval);
        let wav = gen_eval.forward(z_eval, None, false).reshape([1, n]);

        let (target_host, _) = ds.segment_wav(u, 0, t_out);
        let target = Tensor::<2>::from_data(TensorData::new(target_host, [1, n]), device_eval);

        let lv: f64 = mel::mel_l1(wav.clone(), target, cfg, basis)
            .into_scalar::<f32>() as f64;

        let vals = wav.into_data().as_slice::<f32>()?.to_vec();
        let finite = vals.iter().all(|v| v.is_finite());
        let rms = if vals.is_empty() {
            0.0
        } else {
            (vals.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / vals.len() as f64).sqrt()
        };

        out.push(UttMetric {
            utt_id: utt.utt_id.clone(),
            speaker_id: ds.speakers[utt.speaker].clone(),
            mel_l1: lv,
            rms,
            samples: n,
            finite,
        });
    }
    if skipped > 0 {
        eprintln!("⚠️  eval bỏ qua {skipped} câu quá ngắn (< {MIN_SEG} frame latent)");
    }
    Ok(out)
}

/// Gộp chỉ số từng câu thành báo cáo.
pub fn report_from(per_utt: Vec<UttMetric>, drift: f64) -> EvalReport {
    let mean_mel_l1 = if per_utt.is_empty() {
        0.0
    } else {
        per_utt.iter().map(|m| m.mel_l1).sum::<f64>() / per_utt.len() as f64
    };

    // Gom theo giọng.
    let mut acc: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    for m in &per_utt {
        let e = acc.entry(m.speaker_id.clone()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += m.mel_l1;
    }
    let per_speaker: BTreeMap<String, SpeakerStat> = acc
        .into_iter()
        .map(|(k, (count, sum))| {
            (
                k,
                SpeakerStat {
                    count,
                    mean_mel_l1: sum / count as f64,
                },
            )
        })
        .collect();

    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for s in per_speaker.values() {
        lo = lo.min(s.mean_mel_l1);
        hi = hi.max(s.mean_mel_l1);
    }
    let speaker_gap = if per_speaker.is_empty() || mean_mel_l1 <= 0.0 {
        0.0
    } else {
        (hi - lo) / mean_mel_l1
    };

    let min_rms = per_utt.iter().map(|m| m.rms).fold(f64::INFINITY, f64::min);
    let max_rms = per_utt.iter().map(|m| m.rms).fold(f64::NEG_INFINITY, f64::max);
    let all_finite = per_utt.iter().all(|m| m.finite);

    EvalReport {
        per_utt,
        mean_mel_l1,
        per_speaker,
        speaker_gap,
        min_rms: if min_rms.is_finite() { min_rms } else { 0.0 },
        max_rms: if max_rms.is_finite() { max_rms } else { 0.0 },
        all_finite,
        drift,
    }
}

/// Một lượt eval đầy đủ (drift = 0; người gọi chạy lại để đo drift).
pub fn evaluate(
    adapter: &Adapter,
    gen_eval: &Generator,
    ds: &Dataset,
    cfg: &MelConfig,
    basis: &Tensor<2>,
    device: &Device,
    device_eval: &Device,
) -> Result<EvalReport, Box<dyn std::error::Error>> {
    Ok(report_from(
        per_utt_metrics(adapter, gen_eval, ds, cfg, basis, device, device_eval)?,
        0.0,
    ))
}

/// Ngưỡng cổng PASS/FAIL. Đọc từ file `key = value` (tập con của TOML) hoặc mặc định.
#[derive(Debug, Clone, Copy)]
pub struct Gate {
    pub max_mel_l1: f64,
    pub max_speaker_gap: f64,
    pub min_rms: f64,
    pub max_rms: f64,
    pub max_drift: f64,
}

impl Default for Gate {
    fn default() -> Self {
        Self {
            max_mel_l1: 1.0,
            max_speaker_gap: 1.0,
            min_rms: 1e-5,
            max_rms: 1.0,
            max_drift: 1e-6,
        }
    }
}

impl Gate {
    /// Nạp ngưỡng từ file `key = value`; khoá không biết là lỗi (bắt typo).
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("mở gate {} lỗi: {e}", path.display()))?;
        Self::from_str(&text)
    }

    pub fn from_str(text: &str) -> Result<Self, String> {
        let mut g = Gate::default();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let (k, v) = line
                .split_once('=')
                .ok_or_else(|| format!("gate dòng {}: thiếu `=` — `{raw}`", i + 1))?;
            let v: f64 = v
                .trim()
                .parse()
                .map_err(|_| format!("gate dòng {}: giá trị không phải số — `{}`", i + 1, v.trim()))?;
            match k.trim() {
                "max_mel_l1" => g.max_mel_l1 = v,
                "max_speaker_gap" => g.max_speaker_gap = v,
                "min_rms" => g.min_rms = v,
                "max_rms" => g.max_rms = v,
                "max_drift" => g.max_drift = v,
                other => return Err(format!("gate dòng {}: khoá lạ `{other}`", i + 1)),
            }
        }
        Ok(g)
    }

    /// `(pass, danh sách lý do fail)`.
    pub fn check(&self, r: &EvalReport) -> (bool, Vec<String>) {
        let mut fails = Vec::new();
        if !r.all_finite {
            fails.push("có mẫu audio không hữu hạn (NaN/Inf)".to_string());
        }
        if !(r.mean_mel_l1 <= self.max_mel_l1) {
            fails.push(format!(
                "mel_l1 {:.4} > {} (max)",
                r.mean_mel_l1, self.max_mel_l1
            ));
        }
        if !(r.speaker_gap <= self.max_speaker_gap) {
            fails.push(format!(
                "speaker_gap {:.4} > {} (max)",
                r.speaker_gap, self.max_speaker_gap
            ));
        }
        if !(r.min_rms >= self.min_rms) {
            fails.push(format!(
                "rms nhỏ nhất {:.3e} < {} (min)",
                r.min_rms, self.min_rms
            ));
        }
        if !(r.max_rms <= self.max_rms) {
            fails.push(format!(
                "rms lớn nhất {:.3e} > {} (max)",
                r.max_rms, self.max_rms
            ));
        }
        if !(r.drift <= self.max_drift) {
            fails.push(format!(
                "drift {:.3e} > {} (max) — eval không tất định",
                r.drift, self.max_drift
            ));
        }
        (fails.is_empty(), fails)
    }
}

/// So sánh với một baseline (version trước): `Some(lý do)` nếu thoái lui quá
/// `max_rel` (tương đối). Dùng `summary.json` của lần eval trước làm golden.
///
/// Bỏ qua khi giá trị không hữu hạn hoặc baseline ≤ 0 (không có mốc so).
pub fn regression_fail(current: f64, baseline: f64, max_rel: f64) -> Option<String> {
    if !current.is_finite() || !baseline.is_finite() || baseline <= 0.0 {
        return None;
    }
    if current > baseline * (1.0 + max_rel) {
        Some(format!(
            "thoái lui so với baseline: mel_l1 {current:.5} > {:.5} × (1+{max_rel})",
            baseline
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(mel: f64, rms: f64, finite: bool, spk: &str) -> UttMetric {
        UttMetric {
            utt_id: "u".into(),
            speaker_id: spk.into(),
            mel_l1: mel,
            rms,
            samples: 256,
            finite,
        }
    }

    #[test]
    fn gate_parses_and_rejects_unknown_key() {
        let g = Gate::from_str("max_mel_l1 = 0.5\n# note\nmin_rms = 1e-3\n").unwrap();
        assert_eq!(g.max_mel_l1, 0.5);
        assert_eq!(g.min_rms, 1e-3);
        assert!(Gate::from_str("wat = 1\n").unwrap_err().contains("khoá lạ"));
    }

    #[test]
    fn gate_passes_clean_report() {
        let r = report_from(
            vec![dummy(0.2, 0.05, true, "a"), dummy(0.25, 0.06, true, "b")],
            0.0,
        );
        let g = Gate::default();
        let (pass, fails) = g.check(&r);
        assert!(pass, "fails: {fails:?}");
    }

    #[test]
    fn gate_fails_on_nan_and_silence() {
        let r = report_from(
            vec![dummy(0.2, 0.0, false, "a"), dummy(0.2, 0.05, true, "b")],
            0.0,
        );
        let (pass, fails) = Gate::default().check(&r);
        assert!(!pass);
        assert!(fails.iter().any(|f| f.contains("hữu hạn")));
        assert!(fails.iter().any(|f| f.contains("rms nhỏ nhất")));
    }

    #[test]
    fn speaker_gap_is_relative() {
        let r = report_from(
            vec![dummy(0.1, 0.05, true, "a"), dummy(0.3, 0.05, true, "b")],
            0.0,
        );
        // mean 0.2, (0.3-0.1)/0.2 = 1.0
        assert!((r.speaker_gap - 1.0).abs() < 1e-9, "gap {}", r.speaker_gap);
    }

    #[test]
    fn regression_detects_worsening() {
        assert!(regression_fail(0.12, 0.10, 0.05).is_some()); // +20% > 5%
        assert!(regression_fail(0.105, 0.10, 0.05).is_none()); // +5% không vượt
        assert!(regression_fail(0.08, 0.10, 0.05).is_none()); // tốt hơn
        assert!(regression_fail(0.10, 0.0, 0.05).is_none()); // baseline vô nghĩa
        assert!(regression_fail(f64::NAN, 0.1, 0.05).is_none());
    }
}
