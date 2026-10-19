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

/// Chia `[0, total)` thành các chunk kiểu **Applio**: mỗi biên cắt tự tìm điểm
/// **energy min** trong cửa sổ `±W` quanh mốc nominal `s + chunk`
/// (`W = chunk/4`) — biên rơi trúng chỗ nghỉ tự nhiên của giọng đọc.
/// - `chunk == 0` hoặc `total < chunk` → 1 chunk nguyên câu (nếu ≥ MIN_SEG).
/// - Ràng buộc: mỗi chunk ≥ MIN_SEG (đáy STFT/c loss).
/// - Đuôi cuối giữ nếu ≥ MIN_SEG (đáy thật; bỏ rule 50% cũ).
/// - Độ dài chunk biến động `±W` — train/eval pad + mask nên không vấn đề.
/// Tie → cắt sớm nhất (leftmost min, tương đương `np.where(min)` của Applio).
pub fn plan_chunks(total: usize, chunk: usize, energy: &[f32]) -> Vec<(usize, usize)> {
    if chunk == 0 || total < chunk {
        return if total >= MIN_SEG { vec![(0, total)] } else { Vec::new() };
    }
    if total < MIN_SEG {
        return Vec::new();
    }
    let w = (chunk / 4).max(1);
    let mut chunks = Vec::new();
    let mut s = 0usize;
    while total - s > chunk {
        let n = s + chunk;
        // Cắt không vượt mốc nominal (n) → mọi chunk ≤ chunk_frames (batch pad bền vững).
        let lo = (n - w).max(s + MIN_SEG + 1);
        let hi = (n + w).min(total).min(n).min((total - MIN_SEG).max(lo));
        // energy ngoài mảng coi như ∞ → chỉ chọn điểm có dữ liệu.
        let mut cut = None;
        let mut best_e = f32::INFINITY;
        for c in lo..=hi {
            let e = energy.get(c).copied().unwrap_or(f32::INFINITY);
            if e < best_e {
                best_e = e;
                cut = Some(c);
            }
        }
        let Some(c) = cut else { break };
        chunks.push((s, c - s));
        s = c;
    }
    let tail = total - s;
    if tail >= MIN_SEG {
        chunks.push((s, tail));
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
    fn plan_chunks_uniform_energy_full_coverage() {
        // energy đều: mỗi biên cắt ở mép trái window (leftmost min).
        // total=1200, chunk=512, W=128:
        //   cut1 = 512-128 = 384, cut2 = 896-128 = 768, tail = 432 ≥ MIN_SEG
        let total = 1200;
        let chunk = 512;
        let energy = vec![1.0; total];
        let chunks = plan_chunks(total, chunk, &energy);
        assert_eq!(chunks, vec![(0, 384), (384, 384), (768, 432)]);
        // phủ kín nguyên câu, không bỏ frame nào
        assert_eq!(chunks.iter().map(|(_, l)| l).sum::<usize>(), total);
        // mọi chunk ≥ MIN_SEG
        assert!(chunks.iter().all(|(_, l)| *l >= MIN_SEG));
    }

    #[test]
    fn plan_chunks_boundary_at_dip() {
        // có dip ở frame 500 → biên thứ nhất phải rơi đúng 500
        let total = 1100;
        let chunk = 500;
        let energy = energy_with_dip_at(500, total);
        let chunks = plan_chunks(total, chunk, &energy);
        assert_eq!(chunks[0], (0, 500));
        assert_eq!(chunks[1].0, 500);
        // phủ kín
        assert_eq!(chunks.iter().map(|(_, l)| l).sum::<usize>(), total);
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
    fn plan_chunks_total_less_than_chunk_is_full() {
        let energy = vec![1.0; 200];
        let chunks = plan_chunks(200, 512, &energy);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (0, 200));
    }

    #[test]
    fn plan_chunks_total_less_than_min_seg_is_empty() {
        let energy = vec![1.0; 20];
        let chunks = plan_chunks(20, 512, &energy);
        assert!(chunks.is_empty());
    }

    #[test]
    fn plan_chunks_tail_kept_when_over_min_seg() {
        // total = 1400, chunk = 512, W = 128, energy đều → cuts tại 384, 768, 1152;
        // tail (1152, 248) ≥ MIN_SEG → giữ, phủ kín nguyên câu.
        let total = 1400;
        let chunk = 512;
        let energy = vec![1.0; total];
        let chunks = plan_chunks(total, chunk, &energy);
        assert_eq!(chunks, vec![(0, 384), (384, 384), (768, 384), (1152, 248)]);
        assert_eq!(chunks.iter().map(|(_, l)| l).sum::<usize>(), total);
    }

    #[test]
    fn plan_chunks_tail_dropped_below_min_seg() {
        // total = 70 < chunk + w: cắt tại 48, đuôi 22 < MIN_SEG(32) → bị bỏ.
        let total = 70;
        let chunk = 64;
        let energy = vec![1.0; total];
        let chunks = plan_chunks(total, chunk, &energy);
        assert_eq!(chunks, vec![(0, 48)]);
    }

    #[test]
    fn plan_chunks_tie_picks_leftmost() {
        // energy đều → min xuất hiện khắp window → cắt tại mép trái window
        let total = 1000;
        let chunk = 500;
        let energy = vec![1.0; total];
        let chunks = plan_chunks(total, chunk, &energy);
        // cut1 = 500 - 125 = 375
        assert_eq!(chunks[0].1, 375);
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