//! Nạp dataset nhiều giọng cho `train bridge` (§13.6.4f).
//!
//! Manifest CSV ([`crate::manifest`]) trỏ tới wav 16 kHz và features ContentVec
//! `.npy` (`[1, T_h, 768]`, time-major). Ở đây:
//!
//! - nạp wav → resample 22.05 kHz **ngoài mạng** (giống `train_adapter`),
//! - nạp features và hoán vị sang channel-major `[1, 768, T_h]`,
//! - chunking: câu dài → nhiều chunk ≤ `chunk_frames`, biên ở vùng yên (energy minima).
//!
//! Mọi thứ dùng số học xác định (không phụ thuộc thread) để `eval` chạy lại luôn
//! cho cùng kết quả — điều kiện cho chỉ số `drift` = 0 (§13.6.4g).

use std::path::Path;

use crate::manifest::Entry;
use crate::streaming::{energy_profile_wav, plan_chunks};

/// Tần số frame latent của `dec` (`22050 / 256`) — §14.4.
pub const LATENT_HZ: f64 = 86.1328125;
/// Tần số frame HuBERT/ContentVec.
pub const HUBERT_HZ: f64 = 50.0;

/// Số mẫu audio mỗi frame latent.
pub const SAMPLES_PER_FRAME: usize = 256;

/// Một câu gốc: features (time-major) + wav đích 22.05 kHz.
/// Lưu nguyên file, không nhân bản — chunk chỉ lưu offset/len.
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
    /// f0 RMVPE @100 Hz, co-location cạnh feats (`.f0.npy`). `None` nếu chưa có.
    pub f0: Option<Vec<f32>>,
}

/// Một chunk trong câu — đơn vị sample training/eval/generate.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chỉ số vào `Dataset::utts` (câu gốc).
    pub utt: usize,
    /// Frame latent bắt đầu trong câu gốc.
    pub offset_lat: usize,
    /// Độ dài chunk (luôn == `chunk_frames`, đuôi < chunk_frames đã bị bỏ).
    pub len_lat: usize,
    /// ID dạng "utt_id#k" để trace/log.
    pub chunk_id: String,
}

/// Dataset đã nạp, kèm chỉ mục theo giọng ở cấp chunk.
#[derive(Debug)]
pub struct Dataset {
    pub speakers: Vec<String>,
    pub utts: Vec<Utterance>,
    /// `per_speaker[s]` = danh sách chỉ số `utts` thuộc giọng `s` (giữ cho backward compat nếu cần).
    pub per_speaker: Vec<Vec<usize>>,
    /// Danh sách tất cả chunks (đơn vị sample thực tế).
    pub chunks: Vec<Chunk>,
    /// `per_speaker_chunks[s]` = chỉ số chunks thuộc giọng `s`.
    pub per_speaker_chunks: Vec<Vec<usize>>,
}

impl Dataset {
    /// Nạp tất cả câu trong `entries`, explode thành chunks theo `chunk_frames`.
    /// - `chunk_frames == 0`: mỗi câu thành 1 chunk nguyên (offset=0, len=latent_len).
    /// - `chunk_frames > 0`: chia câu theo `plan_chunks`, bỏ đuôi < chunk_frames.
    /// Trả lỗi nếu không còn chunk nào sau khi lọc.
    pub fn load(
        entries: &[Entry],
        chunk_frames: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
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

            let (shape, flat) = voice_vits::npy::read_f32(Path::new(&e.feats_path))?;
            if shape.len() != 3 || shape[0] != 1 {
                return Err(format!("features {} phải [1,T,C], được {shape:?}", e.feats_path).into());
            }
            let (t_h, c) = (shape[1], shape[2]);

            let (wav, sr) = crate::data::read_wav_mono(Path::new(&e.wav_path))?;
            let wav = voice_lib::vc::resample_linear(&wav, sr, 22050);

            // f0 co-location: feats_path ".npy" → ".f0.npy" (CSV hợp đồng không đổi)
            let f0_path = Path::new(&e.feats_path).with_extension("f0.npy");
            let f0 = if f0_path.exists() {
                let (shape, flat) = voice_vits::npy::read_f32(&f0_path)?;
                let _t_f0 = match shape.as_slice() {
                    [1, t] => *t,
                    other => {
                        return Err(format!(
                            "f0 {f0_path:?} phải là [1,T], được {other:?}"
                        )
                        .into())
                    }
                };
                Some(flat)
            } else {
                None
            };

            utts.push(Utterance {
                speaker: spk,
                utt_id: e.utt_id.clone(),
                t_h,
                c,
                feats: flat,
                wav,
                f0,
            });
        }

        // Explode utterances → chunks
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut per_speaker_chunks = vec![Vec::new(); speakers.len()];
        let mut dropped_tail_count = 0usize;

        for (u_idx, utt) in utts.iter().enumerate() {
            let total_latent = Self::latent_len_of_utt(utt);
            let energy = energy_profile_wav(&utt.wav);
            let utt_chunks = plan_chunks(total_latent, chunk_frames, &energy);

            if utt_chunks.is_empty() {
                // Câu quá ngắn (< MIN_SEG hoặc < chunk_frames) → bỏ hoàn toàn
                dropped_tail_count += 1;
                continue;
            }

            for (k, (offset, len)) in utt_chunks.into_iter().enumerate() {
                let chunk_idx = chunks.len();
                let chunk_id = format!("{}#{}", utt.utt_id, k);
                let chunk = Chunk {
                    utt: u_idx,
                    offset_lat: offset,
                    len_lat: len,
                    chunk_id,
                };
                chunks.push(chunk);
                per_speaker_chunks[utt.speaker].push(chunk_idx);
            }
        }

        if dropped_tail_count > 0 {
            eprintln!(
                "⚠️  bỏ {dropped_tail_count} câu quá ngắn (< {} frame latent) khi chunk_frames={}",
                crate::eval::MIN_SEG,
                chunk_frames
            );
        }

        // per_speaker (utterance-level) giữ để compat — nhưng thực tế train/eval dùng per_speaker_chunks
        let mut per_speaker = vec![Vec::new(); speakers.len()];
        for (i, u) in utts.iter().enumerate() {
            per_speaker[u.speaker].push(i);
        }
        for (i, list) in per_speaker.iter().enumerate() {
            if list.is_empty() {
                return Err(format!("giọng {} không có câu nào", speakers[i]).into());
            }
        }

        // Validate: ít nhất 1 chunk cho mỗi giọng active
        for (i, list) in per_speaker_chunks.iter().enumerate() {
            if list.is_empty() {
                return Err(format!(
                    "giọng {} không có chunk nào đủ dài (chunk_frames={})",
                    speakers[i], chunk_frames
                )
                .into());
            }
        }

        Ok(Self {
            speakers,
            utts,
            per_speaker,
            chunks,
            per_speaker_chunks,
        })
    }

    /// Số frame latent của utterance (dùng cho test).
    pub fn latent_len_of_utt(utt: &Utterance) -> usize {
        let by_feat = (utt.t_h as f64 * LATENT_HZ / HUBERT_HZ).floor() as usize;
        let by_wav = utt.wav.len() / SAMPLES_PER_FRAME;
        by_feat.min(by_wav).max(1)
    }

    pub fn num_speakers(&self) -> usize {
        self.speakers.len()
    }

    /// Số frame HuBERT cần cho `t_out` frame latent (làm tròn).
    pub fn hubert_frames_for(t_out: usize) -> usize {
        ((t_out as f64 * HUBERT_HZ / LATENT_HZ).round() as usize).max(1)
    }

    /// Số frame latent của chunk `k` (luôn == chunk_frames, trừ khi chunk_frames=0).
    pub fn latent_len(&self, k: usize) -> usize {
        self.chunks[k].len_lat
    }

    /// Cắt `t_h_seg` frame HuBERT từ `h0` của chunk `k`, trả channel-major `[1, c, t_h_seg]`.
    /// `h0` là offset HuBERT *trong chunk* (0 = đầu chunk).
    pub fn segment_feats_channel_major(&self, k: usize, h0: usize, t_h_seg: usize) -> Vec<f32> {
        let chunk = &self.chunks[k];
        let utt = &self.utts[chunk.utt];
        // Chuyển offset latent của chunk → offset HuBERT
        let h0_global = ((chunk.offset_lat as f64 * HUBERT_HZ / LATENT_HZ).floor() as usize) + h0;
        let h1 = (h0_global + t_h_seg).min(utt.t_h);
        let h0_clamped = h0_global.min(h1.saturating_sub(1));
        let t = h1 - h0_clamped;
        let mut out = vec![0.0f32; utt.c * t];
        for (row, feat_row) in (h0_clamped..h1).enumerate() {
            let src = &utt.feats[feat_row * utt.c..(feat_row + 1) * utt.c];
            for (ch, &v) in src.iter().enumerate() {
                out[ch * t + row] = v;
            }
        }
        out
    }

    /// Cắt `t_out` frame latent từ wav đích của chunk `k`.
    /// `l0` là offset latent *trong chunk* (0 = đầu chunk).
    /// Trả `(mẫu, n)` với `n = t_out * 256`.
    pub fn segment_wav(&self, k: usize, l0: usize, t_out: usize) -> (Vec<f32>, usize) {
        let chunk = &self.chunks[k];
        let utt = &self.utts[chunk.utt];
        let n = t_out * SAMPLES_PER_FRAME;
        let global_l0 = chunk.offset_lat + l0;
        let start = global_l0 * SAMPLES_PER_FRAME;
        let end = (start + n).min(utt.wav.len());
        let mut out = utt.wav[start.min(utt.wav.len())..end].to_vec();
        out.resize(n, 0.0);
        (out, n)
    }

    /// Lấy features channel-major của chunk `k` (toàn bộ chunk, dùng cho eval).
    /// `t_out` phải == `latent_len(k)`.
    pub fn chunk_feats_channel_major(&self, k: usize, t_out: usize) -> Vec<f32> {
        let t_h = Self::hubert_frames_for(t_out).min(self.utts[self.chunks[k].utt].t_h);
        self.segment_feats_channel_major(k, 0, t_h)
    }

    /// Kiểm tra tất cả câu gốc cùng chiều features (bắt buộc để xếp batch).
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

    /// Số chunks tổng.
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
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

    fn make_utt(t_h: usize, c: usize, latent_frames: usize) -> Utterance {
        Utterance {
            speaker: 0,
            utt_id: "u".into(),
            t_h,
            c,
            feats: vec![0.5f32; t_h * c],
            wav: vec![0.1f32; latent_frames * SAMPLES_PER_FRAME],
            f0: None,
        }
    }

    #[test]
    fn hubert_frames_round_trip() {
        assert_eq!(Dataset::hubert_frames_for(86), 50);
        assert_eq!(Dataset::hubert_frames_for(1), 1);
    }

    #[test]
    fn chunking_zero_is_full_utt() {
        // chunk_frames = 0 → 1 chunk = cả câu
        let utt = make_utt(100, 768, 200); // ~200 frame latent
        let ds = Dataset {
            speakers: vec!["spk".into()],
            utts: vec![utt],
            per_speaker: vec![vec![0]],
            chunks: vec![Chunk {
                utt: 0,
                offset_lat: 0,
                len_lat: 200,
                chunk_id: "u#0".into(),
            }],
            per_speaker_chunks: vec![vec![0]],
        };
        assert_eq!(ds.num_chunks(), 1);
        assert_eq!(ds.latent_len(0), 200);
        assert_eq!(ds.chunks[0].offset_lat, 0);
    }

    #[test]
    fn chunking_splits_long_utt() {
        // chunk_frames = 512, câu 1200 latent → 2 chunk 512, đuôi 176 bỏ
        let utt = make_utt(600, 768, 1200);
        let ds = Dataset {
            speakers: vec!["spk".into()],
            utts: vec![utt],
            per_speaker: vec![vec![0]],
            chunks: vec![
                Chunk { utt: 0, offset_lat: 0, len_lat: 512, chunk_id: "u#0".into() },
                Chunk { utt: 0, offset_lat: 512, len_lat: 512, chunk_id: "u#1".into() },
            ],
            per_speaker_chunks: vec![vec![0, 1]],
        };
        assert_eq!(ds.num_chunks(), 2);
        assert_eq!(ds.latent_len(0), 512);
        assert_eq!(ds.latent_len(1), 512);
        assert_eq!(ds.chunks[0].offset_lat, 0);
        assert_eq!(ds.chunks[1].offset_lat, 512);
    }

    #[test]
    fn segment_feats_channel_major_chunk() {
        // chunk offset_lat=100, len=512, h0=0 → global h0 = floor(100 * 50 / 86.13) = 58
        let c = 2;
        let t_h = 100;
        let utt = Utterance {
            speaker: 0,
            utt_id: "u".into(),
            t_h,
            c,
            feats: (0..t_h * c).map(|i| i as f32).collect(),
            wav: vec![0.0; 1000 * SAMPLES_PER_FRAME],
            f0: None,
        };
        let ds = Dataset {
            speakers: vec!["spk".into()],
            utts: vec![utt],
            per_speaker: vec![vec![0]],
            chunks: vec![Chunk {
                utt: 0,
                offset_lat: 100,
                len_lat: 512,
                chunk_id: "u#0".into(),
            }],
            per_speaker_chunks: vec![vec![0]],
        };
        // Lấy 2 frame HuBERT từ đầu chunk
        let seg = ds.segment_feats_channel_major(0, 0, 2);
        // global h0 = floor(100 * 50 / 86.13) = 58
        // frame 58: ch0=58*2=116, ch1=117; frame 59: ch0=118, ch1=119
        assert_eq!(seg, vec![116.0, 118.0, 117.0, 119.0]);
    }

    #[test]
    fn segment_wav_chunk() {
        let utt = make_utt(100, 768, 200);
        let ds = Dataset {
            speakers: vec!["spk".into()],
            utts: vec![utt],
            per_speaker: vec![vec![0]],
            chunks: vec![Chunk {
                utt: 0,
                offset_lat: 50,
                len_lat: 100,
                chunk_id: "u#0".into(),
            }],
            per_speaker_chunks: vec![vec![0]],
        };
        // chunk offset=50, l0=0, t_out=10 → global start = 60
        let (wav, n) = ds.segment_wav(0, 0, 10);
        assert_eq!(n, 10 * SAMPLES_PER_FRAME);
        assert_eq!(wav.len(), n);
    }

    #[test]
    fn chunk_feats_channel_major() {
        let utt = make_utt(100, 768, 200);
        let ds = Dataset {
            speakers: vec!["spk".into()],
            utts: vec![utt],
            per_speaker: vec![vec![0]],
            chunks: vec![Chunk {
                utt: 0,
                offset_lat: 0,
                len_lat: 150,
                chunk_id: "u#0".into(),
            }],
            per_speaker_chunks: vec![vec![0]],
        };
        let feats = ds.chunk_feats_channel_major(0, 150);
        let t_h = Dataset::hubert_frames_for(150).min(100);
        assert_eq!(feats.len(), 768 * t_h);
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