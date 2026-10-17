//! Allocating a whole-file frame count across segments — without drift (§14.4).
//!
//! Cutting a long file into segments and rounding each segment's duration
//! independently accumulates error: 600 s cut into 60×10 s drifts by ~18 latent
//! frames (~0.21 s) at 86.13 Hz. The fix is to compute `T_total` **once** from the
//! full duration and split it proportionally with largest-remainder, so the parts
//! always sum back to the whole.

/// Latent frame count for a duration: `round(duration_s × frame_rate)`.
///
/// This is the **only** place the total should be produced; every segment length
/// then comes from [`split_frames`] instead of its own rounding.
pub fn frame_count(duration_s: f64, frame_rate: f64) -> usize {
    if !duration_s.is_finite() || !frame_rate.is_finite() {
        return 0;
    }
    if duration_s <= 0.0 || frame_rate <= 0.0 {
        return 0;
    }
    (duration_s * frame_rate).round() as usize
}

/// Split `total` frames across `weights` with largest-remainder.
///
/// Guarantees `result.iter().sum::<usize>() == total` whenever `total` is split at
/// all. Ties in the fractional part go to the lower index.
///
/// Degenerate inputs:
/// - empty `weights` → empty result;
/// - non-finite or non-positive weights are treated as `0`;
/// - if every weight is zero, the total is spread as evenly as possible.
pub fn split_frames(total: usize, weights: &[f64]) -> Vec<usize> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }

    let clean: Vec<f64> = weights
        .iter()
        .map(|w| if w.is_finite() && *w > 0.0 { *w } else { 0.0 })
        .collect();
    let sum: f64 = clean.iter().sum();

    if sum <= 0.0 {
        // Even spread, remainder to the lowest indices.
        let base = total / n;
        let rem = total % n;
        return (0..n).map(|i| base + usize::from(i < rem)).collect();
    }

    let mut out = vec![0usize; n];
    let mut remainders: Vec<(f64, usize)> = Vec::with_capacity(n);
    let mut assigned = 0usize;

    for (i, w) in clean.iter().enumerate() {
        let quota = total as f64 * w / sum;
        let floor = quota.floor();
        out[i] = floor as usize;
        assigned += out[i];
        remainders.push((quota - floor, i));
    }

    let left = total.saturating_sub(assigned);
    remainders.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    for &(_, idx) in remainders.iter().take(left) {
        out[idx] += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const LATENT_RATE: f64 = 22050.0 / 256.0; // 86.1328125

    #[test]
    fn frame_count_rounds() {
        assert_eq!(frame_count(6.84, LATENT_RATE), 589); // 589.148 → 589
        assert_eq!(frame_count(1.0, LATENT_RATE), 86); // 86.133 → 86
        assert_eq!(frame_count(0.0, LATENT_RATE), 0);
        assert_eq!(frame_count(-1.0, LATENT_RATE), 0);
        assert_eq!(frame_count(1.0, 0.0), 0);
    }

    #[test]
    fn no_drift_when_splitting_600_seconds() {
        // §14.4 worked example: 600 s → 60×10 s segments.
        let total = frame_count(600.0, LATENT_RATE);
        assert_eq!(total, 51680); // 600 × 86.1328125 = 51679.6875

        let weights = vec![1.0; 60];
        let parts = split_frames(total, &weights);

        assert_eq!(parts.iter().sum::<usize>(), total);
        // Near-even: 51680 / 60 = 861.33 → 861 or 862.
        assert!(parts.iter().all(|&p| p == 861 || p == 862));
    }

    #[test]
    fn naive_rounding_would_drift() {
        // Same 600 s split into 60 parts, each rounded independently: drifts.
        let naive: usize = (0..60)
            .map(|_| frame_count(10.0, LATENT_RATE))
            .sum();
        let correct = frame_count(600.0, LATENT_RATE);
        assert_ne!(naive, correct, "the drift this module exists to prevent");
        assert_eq!(naive, 60 * 861); // 861 × 60 = 51660, short of 51680
        assert_eq!(correct - naive, 20); // ≈ 0.23 s of drift over 10 minutes
    }

    #[test]
    fn exact_sum_for_arbitrary_weights() {
        for weights in [
            vec![1.0, 1.0, 1.0],
            vec![3.0, 1.0, 1.0],
            vec![0.5, 0.25, 0.25],
            vec![10.0, 1.0],
        ] {
            for total in [0, 1, 7, 100, 589, 51679] {
                let parts = split_frames(total, &weights);
                assert_eq!(parts.iter().sum::<usize>(), total, "total={total}");
            }
        }
    }

    #[test]
    fn zero_total_yields_zeros() {
        assert_eq!(split_frames(0, &[1.0, 2.0]), vec![0, 0]);
    }

    #[test]
    fn empty_weights_yield_empty() {
        assert!(split_frames(100, &[]).is_empty());
    }

    #[test]
    fn all_zero_weights_spread_evenly() {
        assert_eq!(split_frames(10, &[0.0, 0.0, 0.0]), vec![4, 3, 3]);
        assert_eq!(split_frames(9, &[0.0, 0.0, 0.0]), vec![3, 3, 3]);
    }

    #[test]
    fn non_positive_and_non_finite_weights_get_nothing() {
        let parts = split_frames(10, &[1.0, -1.0, f64::NAN, f64::INFINITY]);
        assert_eq!(parts, vec![10, 0, 0, 0]);
    }

    #[test]
    fn ties_go_to_the_lower_index() {
        // 10 / 3 → 3,3,3 with one remainder → first index wins.
        assert_eq!(split_frames(10, &[1.0, 1.0, 1.0]), vec![4, 3, 3]);
    }

    #[test]
    fn proportional_weights_are_respected() {
        let parts = split_frames(100, &[3.0, 1.0]);
        assert_eq!(parts.iter().sum::<usize>(), 100);
        assert!((parts[0] as f64 - 75.0).abs() <= 1.0);
        assert!((parts[1] as f64 - 25.0).abs() <= 1.0);
    }
}
