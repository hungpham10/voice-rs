//! Nạp dataset nhiều giọng cho `train bridge` (§13.6.4f).
//!
//! Manifest CSV ([`crate::manifest`]) trỏ tới wav 16 kHz và features ContentVec
//! `.npy` (`[1, T_h, 768]`, time-major). Ở đây:
//!
//! - nạp wav → resample 22.05 kHz **ngoài mạng** (giống `train_adapter`),
//! - nạp features và hoán vị sang channel-major `[1, 768, T_h]`,
//! - cắt đoạn (segment) **cân bằng theo giọng** ở tầng trên.
//!
//! Mọi thứ dùng số học xác định (không phụ thuộc thread) để `eval` chạy lại luôn
//! cho cùng kết quả — điều kiện cho chỉ số `drift` = 0 (§13.6.4g).

use std::path::Path;

use crate::manifest::Entry;

/// Tần số frame latent của `dec` (`22050 / 256`) — §14.4.
pub const LATENT_HZ: f64 = 86.1328125;
/// Tần số frame HuBERT/ContentVec.
pub const HUBERT_HZ: f64 = 50.0;

/// Số mẫu audio mỗi frame latent.
pub const SAMPLES_PER_FRAME: usize = 256;

/// Một câu: features (time-major) + wav đích 22.05 kHz.
#[derive(Debug, Clone)]
pub struct Utterance {
    /// Chỉ số giọng trong [`Dataset::speakers`].
    pub speaker: usize,
    pub utt_id: String,
    /// Số frame HuBERT.
    pub t_h: usize,
    /// Chiều features (768).
    pub c: usize,
    /// Time-major `[t_h, c]`.
    pub feats: Vec<f32>,
    /// Wav đích đã resample 22.05 kHz.
    pub wav: Vec<f32>,
}

/// Dataset đã nạp, kèm chỉ mục theo giọng.
#[derive(Debug)]
pub struct Dataset {
    pub speakers: Vec<String>,
    pub utts: Vec<Utterance>,
    /// `per_speaker[s]` = danh sách chỉ số `utts` thuộc giọng `s`.
    pub per_speaker: Vec<Vec<usize>>,
}

impl Dataset {
    /// Nạp tất cả câu trong `entries` (đã lọc theo split bởi người gọi).
    pub fn load(entries: &[Entry]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut speakers: Vec<String> = Vec::new();
        let mut utts: Vec<Utterance> = Vec::new();

        for e in entries {
            if !e.has_wav() {
                return Err(format!("{} chưa có (wav)", e.wav_path).into());
            }
            if !e.has_features() {
                return Err(format!(
                    "{} chưa có — chạy `embed` trước",
                    e.feats_path
                )
                .into());
            }
            let spk = match speakers.iter().position(|s| s == &e.speaker_id) {
                Some(i) => i,
                None => {
                    speakers.push(e.speaker_id.clone());
                    speakers.len() - 1
                }
            };

            let (shape, flat) = voice_models::npy::read_f32(Path::new(&e.feats_path))?;
            if shape.len() != 3 || shape[0] != 1 {
                return Err(format!("features {} phải [1,T,C], được {shape:?}", e.feats_path).into());
            }
            let (t_h, c) = (shape[1], shape[2]);

            let (wav, sr) = crate::data::read_wav_mono(Path::new(&e.wav_path))?;
            let wav = voice_lib::vc::resample_linear(&wav, sr, 22050);

            utts.push(Utterance {
                speaker: spk,
                utt_id: e.utt_id.clone(),
                t_h,
                c,
                feats: flat,
                wav,
            });
        }

        let mut per_speaker = vec![Vec::new(); speakers.len()];
        for (i, u) in utts.iter().enumerate() {
            per_speaker[u.speaker].push(i);
        }
        // Giọng không có câu nào là lỗi cấu hình — báo sớm.
        for (i, list) in per_speaker.iter().enumerate() {
            if list.is_empty() {
                return Err(format!("giọng {} không có câu nào", speakers[i]).into());
            }
        }

        Ok(Self { speakers, utts, per_speaker })
    }

    pub fn num_speakers(&self) -> usize {
        self.speakers.len()
    }

    /// Số frame HuBERT cần cho `t_out` frame latent (làm tròn).
    pub fn hubert_frames_for(t_out: usize) -> usize {
        ((t_out as f64 * HUBERT_HZ / LATENT_HZ).round() as usize).max(1)
    }

    /// Số frame latent tối đa sinh được từ câu `u` (giới hạn bởi cả features và wav).
    pub fn latent_len(&self, u: usize) -> usize {
        let utt = &self.utts[u];
        let by_feat = (utt.t_h as f64 * LATENT_HZ / HUBERT_HZ).floor() as usize;
        let by_wav = utt.wav.len() / SAMPLES_PER_FRAME;
        by_feat.min(by_wav).max(1)
    }

    /// Cắt `t_h_seg` frame HuBERT từ `h0`, trả **channel-major** `[1, c, t_h_seg]`.
    pub fn segment_feats_channel_major(&self, u: usize, h0: usize, t_h_seg: usize) -> Vec<f32> {
        let utt = &self.utts[u];
        let h1 = (h0 + t_h_seg).min(utt.t_h);
        let h0 = h0.min(h1.saturating_sub(1));
        let t = h1 - h0;
        let mut out = vec![0.0f32; utt.c * t];
        for (row, feat_row) in (h0..h1).enumerate() {
            let src = &utt.feats[feat_row * utt.c..(feat_row + 1) * utt.c];
            for (ch, &v) in src.iter().enumerate() {
                out[ch * t + row] = v;
            }
        }
        out
    }

    /// Cắt `t_out` frame latent (bắt đầu ở latent `l0`) từ wav đích.
    /// Trả `(mẫu, n)` với `n = t_out * 256` (đã kẹp theo wav thực có).
    pub fn segment_wav(&self, u: usize, l0: usize, t_out: usize) -> (Vec<f32>, usize) {
        let utt = &self.utts[u];
        let n = t_out * SAMPLES_PER_FRAME;
        let start = l0 * SAMPLES_PER_FRAME;
        let end = (start + n).min(utt.wav.len());
        let mut out = utt.wav[start.min(utt.wav.len())..end].to_vec();
        // Kẹp lại cho đúng `n` để batch đều (pad bằng 0 nếu thiếu — hiếm khi xảy ra).
        out.resize(n, 0.0);
        (out, n)
    }

    /// Toàn bộ câu, channel-major, dùng cho `eval` (xác định, đủ dài).
    pub fn full_feats_channel_major(&self, u: usize, t_out: usize) -> Vec<f32> {
        let t_h = Self::hubert_frames_for(t_out).min(self.utts[u].t_h);
        self.segment_feats_channel_major(u, 0, t_h)
    }

    /// Kiểm tra tất cả câu cùng chiều features (bắt buộc để xếp batch).
    pub fn check_dims(&self) -> Result<usize, String> {
        let mut c = None;
        for u in &self.utts {
            match c {
                None => c = Some(u.c),
                Some(v) if v != u.c => {
                    return Err(format!("features không cùng chiều: {v} vs {}", u.c))
                }
                _ => {}
            }
        }
        c.ok_or_else(|| "dataset rỗng".to_string())
    }
}

/// Bộ sinh số giả ngẫu nhiên xác định (xorshift64*) — không thêm dependency, và
/// cùng seed luôn cho cùng dãy (`drift` = 0 ở `eval`).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Tránh state 0 (xorshift chết ở 0).
        Self(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Số nguyên trong `[0, n)`; trả 0 nếu `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// Tensor batch helper: ghép nhiều `[1, C, T]` channel-major thành `[B, C, T]`.
pub fn stack_channel_major(segments: &[Vec<f32>], c: usize, t: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(segments.len() * c * t);
    for seg in segments {
        out.extend_from_slice(seg);
    }
    out
}

/// Tensor wav batch `[B, N]` từ các đoạn `[1, n]`.
pub fn stack_rows(rows: &[(Vec<f32>, usize)]) -> (Vec<f32>, usize, usize) {
    let n = rows.first().map(|r| r.1).unwrap_or(0);
    let mut out = Vec::with_capacity(rows.len() * n);
    for (v, _) in rows {
        out.extend_from_slice(v);
    }
    (out, rows.len(), n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hubert_frames_round_trip() {
        // 86 latent frame ≈ 1 s → ~50 frame HuBERT.
        assert_eq!(Dataset::hubert_frames_for(86), 50);
        assert_eq!(Dataset::hubert_frames_for(1), 1);
    }

    fn fake_utt() -> Utterance {
        // 4 frame HuBERT, c = 2 → channel-major [1,2,4].
        Utterance {
            speaker: 0,
            utt_id: "u".into(),
            t_h: 4,
            c: 2,
            feats: vec![
                1.0, 2.0, // frame 0: ch0, ch1
                3.0, 4.0, // frame 1
                5.0, 6.0, // frame 2
                7.0, 8.0, // frame 3
            ],
            wav: vec![0.0; SAMPLES_PER_FRAME * 10],
        }
    }

    #[test]
    fn channel_major_slicing_is_correct() {
        let ds = Dataset {
            speakers: vec!["a".into()],
            utts: vec![fake_utt()],
            per_speaker: vec![vec![0]],
        };
        // Lấy frame 1..3 → channel-major: ch0 = [3,5], ch1 = [4,6].
        let seg = ds.segment_feats_channel_major(0, 1, 2);
        assert_eq!(seg, vec![3.0, 5.0, 4.0, 6.0]);
    }

    #[test]
    fn latent_len_limited_by_features() {
        let ds = Dataset {
            speakers: vec!["a".into()],
            utts: vec![fake_utt()],
            per_speaker: vec![vec![0]],
        };
        // 4 frame HuBERT → floor(4 * 86.13/50) = 6 latent; wav cho 10 → min = 6.
        assert_eq!(ds.latent_len(0), 6);
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn rng_below_stays_in_range() {
        let mut r = Rng::new(1);
        for _ in 0..1000 {
            assert!(r.below(7) < 7);
        }
        assert_eq!(r.below(0), 0);
    }
}
