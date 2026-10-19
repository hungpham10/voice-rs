//! Streaming chunking: chia câu dài thành các chunk bằng nhau (≤ chunk_frames),
//! biên cắt dời về vùng yên tĩnh (energy minima) để ghép mượt không cần crossfade.
//! Dùng chung cho train/eval/generate.

use crate::dataset::{HUBERT_HZ, LATENT_HZ, SAMPLES_PER_FRAME};
use crate::eval::MIN_SEG;

/// Năng lượng nhỏ coi là "im lặng" (~−45 dBFS RMS).
pub const SIL_RMS: f32 = 0.00562341325; // 10^(-45/20)

/// Peak normalize như Applio offline.
pub const PEAK: f32 = 0.99;

/// Profil năng lượng theo frame latent từ wav 22.05 kHz.
/// Mỗi frame = SAMPLES_PER_FRAME (256) mẫu → RMS cửa sổ.
pub fn energy_profile_wav(wav: &[f32]) -> Vec<f32> {
    if wav.is_empty() {
        return Vec::new();
    }
    let n_chunks = (wav.len() + SAMPLES_PER_FRAME - 1) / SAMPLES_PER_FRAME;
    let mut energy = Vec::with_capacity(n_chunks);
    for i in 0..n_chunks {
        let start = i * SAMPLES_PER_FRAME;
        let end = (start + SAMPLES_PER_FRAME).min(wav.len());
        let slice = &wav[start..end];
        let sum_sq: f32 = slice.iter().map(|v| *v * *v).sum();
        let rms = (sum_sq / slice.len() as f32).sqrt();
        energy.push(rms);
    }
    energy
}

/// Profil năng lượng proxy từ features ContentVec (channel-major `[1, c, t_h]`).
/// Tính L2 norm theo frame HuBERT, sau đó nội suy/đánh chỉ mục sang frame latent.
/// Dùng cho render (không có wav).
pub fn energy_profile_feats(feats_cm: &[f32], t_h: usize, c: usize) -> Vec<f32> {
    if t_h == 0 || c == 0 {
        return Vec::new();
    }
    // L2 norm mỗi frame HuBERT: sqrt(sum_c feats[ch * t_h + t]^2)
    let mut hubert_energy = Vec::with_capacity(t_h);
    for t in 0..t_h {
        let mut sum_sq = 0.0f32;
        for ch in 0..c {
            let v = feats_cm[ch * t_h + t];
            sum_sq += v * v;
        }
        hubert_energy.push(sum_sq.sqrt());
    }
    // Ánh xạ sang frame latent: mỗi latent frame t_out ↔ t_h = round(t_out * HUBERT_HZ / LATENT_HZ)
    // Ta dùng nearest (floor) để mapping nhanh, đủ tốt cho việc tìm cực tiểu.
    let t_out_max = ((t_h as f64 * LATENT_HZ / HUBERT_HZ).floor() as usize).max(1);
    let mut energy = Vec::with_capacity(t_out_max);
    for t_out in 0..t_out_max {
        let t_h_idx = ((t_out as f64 * HUBERT_HZ / LATENT_HZ).round() as usize).min(t_h - 1);
        energy.push(hubert_energy[t_h_idx]);
    }
    energy
}

/// Chọn offset toàn cục `o ∈ [0, grid)` sao cho tổng energy tại các điểm cắt
/// `{o + k*chunk | k = 1..n_chunks-1}` là nhỏ nhất.
/// grid = max(1, chunk / 2). Tie → offset nhỏ nhất (deterministic).
fn best_offset(total: usize, chunk: usize, energy: &[f32]) -> usize {
    if chunk == 0 || total <= chunk {
        return 0;
    }
    let grid = (chunk / 2).max(1);
    let n_chunks = total / chunk;
    if n_chunks <= 1 {
        return 0;
    }
    let mut best_o = 0;
    let mut best_sum = f32::INFINITY;
    for o in 0..grid {
        let mut sum = 0.0f32;
        for k in 1..n_chunks {
            let cut = o + k * chunk;
            if cut < energy.len() {
                sum += energy[cut];
            }
        }
        if sum < best_sum {
            best_sum = sum;
            best_o = o;
        }
    }
    best_o
}

/// Chia `[0, total)` thành các chunk dài đúng `chunk` frame.
/// - `chunk == 0` → 1 chunk `(0, total)` (cả câu).
/// - `total < chunk` → rỗng.
/// - Chỉ trả về chunk có `len == chunk` (đuôi < chunk bị bỏ).
/// - Biên cắt = `best_offset(total, chunk, energy) + k*chunk`.
pub fn plan_chunks(total: usize, chunk: usize, energy: &[f32]) -> Vec<(usize, usize)> {
    if chunk == 0 {
        return if total >= MIN_SEG {
            vec![(0, total)]
        } else {
            Vec::new()
        };
    }
    if total < chunk || total < MIN_SEG {
        return Vec::new();
    }
    let n_chunks = total / chunk;
    if n_chunks == 0 {
        return Vec::new();
    }
    let o = best_offset(total, chunk, energy);
    let mut chunks = Vec::with_capacity(n_chunks);
    for k in 0..n_chunks {
        let start = o + k * chunk;
        let end = start + chunk;
        if end <= total {
            chunks.push((start, chunk));
        }
    }
    chunks
}

/// Kế hoạch render 1 chunk kèm context, tính bằng frame latent.
/// Input: `(chunk_start, chunk_len)` từ `plan_chunks`, tổng `t_out` của cả câu,
/// context frames `ctx` (±ctx hai bên).
/// Output: `(feat_start, feat_len, trim_left)` —
/// - `feat_start`: frame latent bắt đầu slice feats (có context trái)
/// - `feat_len`: tổng frame latent lấy cho gen (chunk_len + ctx_trái + ctx_phải)
/// - `trim_left`: số frame latent bỏ ở đầu wav output (ctx_trái thực tế sau clamp).
/// Ràng buộc: `feat_start >= 0`, `feat_start + feat_len <= t_out`.
pub fn chunk_ctx_span(
    chunk: (usize, usize),
    t_out: usize,
    ctx: usize,
) -> (usize, usize, usize) {
    let (c_start, c_len) = chunk;
    let ctx_left = ctx.min(c_start);
    let ctx_right = ctx.min(t_out.saturating_sub(c_start + c_len));
    let feat_start = c_start - ctx_left;
    let feat_len = c_len + ctx_left + ctx_right;
    (feat_start, feat_len, ctx_left)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn energy_with_dip_at(idx: usize, len: usize) -> Vec<f32> {
        let mut e = vec![1.0; len];
        if idx < len {
            e[idx] = 0.01;
        }
        e
    }

    #[test]
    fn plan_chunks_basic_uniform() {
        let total = 1200;
        let chunk = 512;
        // energy đều → offset = 0
        let energy = vec![1.0; total];
        let chunks = plan_chunks(total, chunk, &energy);
        // 1200 / 512 = 2 chunks full (1024), đuôi 176 bị bỏ
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], (0, 512));
        assert_eq!(chunks[1], (512, 512));
        // mọi chunk đúng 512
        for (_, len) in &chunks {
            assert_eq!(*len, 512);
        }
    }

    #[test]
    fn plan_chunks_offset_at_dip() {
        let total = 1100;
        let chunk = 500;
        // có dip ở frame 500 → offset nên chọn gần đó
        let energy = energy_with_dip_at(500, total);
        let chunks = plan_chunks(total, chunk, &energy);
        // 1100 / 500 = 2 chunks (1000), đuôi 100 bỏ
        assert_eq!(chunks.len(), 2);
        // offset nên là 0 (gần dip 500 nhất trong grid [0, 250))
        // vì dip ở 500 = 1*500 + 0, offset=0 cho cut tại 500
        assert_eq!(chunks[0].0, 0);
        assert_eq!(chunks[1].0, 500);
    }

    #[test]
    fn plan_chunks_chunk_zero_is_full() {
        let total = 1000;
        let energy = vec![1.0; total];
        let chunks = plan_chunks(total, 0, &energy);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (0, total));
    }

    #[test]
    fn plan_chunks_total_less_than_chunk_is_empty() {
        let energy = vec![1.0; 200];
        let chunks = plan_chunks(200, 512, &energy);
        assert!(chunks.is_empty());
    }

    #[test]
    fn plan_chunks_total_less_than_min_seg_is_empty() {
        let energy = vec![1.0; 20];
        let chunks = plan_chunks(20, 512, &energy);
        assert!(chunks.is_empty());
    }

    #[test]
    fn best_offset_tie_chooses_smaller() {
        // 2 offset cho sum bằng nhau → chọn offset nhỏ hơn
        let total = 1000;
        let chunk = 500;
        // energy đều 1.0 → mọi offset sum = n_chunks-1 = 1.0
        let energy = vec![1.0; total];
        let chunks = plan_chunks(total, chunk, &energy);
        // offset = 0
        assert_eq!(chunks[0].0, 0);
    }

    #[test]
    fn chunk_ctx_span_no_clamp() {
        // chunk ở giữa, đủ context 2 bên
        let (fs, fl, tl) = chunk_ctx_span((512, 512), 2000, 32);
        assert_eq!((fs, fl, tl), (480, 576, 32));
    }

    #[test]
    fn chunk_ctx_span_clamp_left() {
        // chunk đầu, không đủ context trái
        let (fs, fl, tl) = chunk_ctx_span((0, 512), 2000, 32);
        assert_eq!((fs, fl, tl), (0, 544, 0));
    }

    #[test]
    fn chunk_ctx_span_clamp_right() {
        // chunk cuối, không đủ context phải (chunk kết thúc ở t_out)
        // chunk (1500, 512) với t_out=2000 → c_start + c_len = 2012 > 2000 (overflow)
        // trong thực tế plan_chunks không bao giờ tạo chunk overflow
        // test case hợp lệ: chunk (1488, 512) → end = 2000
        let (fs, fl, tl) = chunk_ctx_span((1488, 512), 2000, 32);
        // ctx_left = 32, ctx_right = 0 (đã hết t_out)
        assert_eq!((fs, fl, tl), (1456, 544, 32));
    }

    #[test]
    fn chunk_ctx_span_ctx_zero() {
        let (fs, fl, tl) = chunk_ctx_span((100, 200), 1000, 0);
        assert_eq!((fs, fl, tl), (100, 200, 0));
    }

    #[test]
    fn energy_profile_wav_empty() {
        assert!(energy_profile_wav(&[]).is_empty());
    }

    #[test]
    fn energy_profile_wav_short() {
        // wav < 1 frame → 1 energy entry
        let wav = vec![0.1f32; 100];
        let e = energy_profile_wav(&wav);
        assert_eq!(e.len(), 1);
    }

    #[test]
    fn energy_profile_feats_empty() {
        assert!(energy_profile_feats(&[], 0, 768).is_empty());
    }
}