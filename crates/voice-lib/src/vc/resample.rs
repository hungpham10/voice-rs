//! Nearest-neighbour frame mapping between two frame rates (§13.2.1, option (a)).
//!
//! This is the "interpolate outside the network" half: no parameters, it only
//! decides which source frame each output frame copies. The learnable part
//! (`Linear`, later) sits after this and does not change the mapping.
//!
//! `nearest` here matches `F.interpolate(mode="nearest")` for an integer scale: for
//! `t_out = 2·t_in` it yields `[0,0,1,1,…]`, i.e. `repeat_interleave(2)` (§13.2).

/// Map each of `t_out` output frames to a source frame in `[0, t_in)`.
///
/// `index(t) = floor(t · t_in / t_out)`, clamped to `t_in - 1`. Returns an empty
/// vector if either side is zero.
///
/// Properties relied on by the pipeline:
/// - non-decreasing;
/// - always in bounds;
/// - when `t_out >= t_in`, no source frame is skipped and the first/last output
///   map to the first/last source frame.
pub fn nearest_index_map(t_in: usize, t_out: usize) -> Vec<usize> {
    if t_in == 0 || t_out == 0 {
        return Vec::new();
    }

    (0..t_out)
        .map(|t| ((t as u128 * t_in as u128) / t_out as u128) as usize)
        .map(|i| i.min(t_in - 1))
        .collect()
}

/// Linear-interpolation sample-rate conversion for mono `f32` PCM.
///
/// Used to bring the 16 kHz target audio up to the 22.05 kHz rate the loss is
/// computed at. It stays **outside** the network (like [`nearest_index_map`]) —
/// only the target-side audio is converted; the generated waveform is already at
/// 22.05 kHz.
///
/// `out_len = round(n_in · to_hz / from_hz)`. Output sample `j` reads source
/// position `j · from_hz / to_hz` and blends the two neighbours; the first and
/// last output samples map to the first/last source samples.
///
/// Linear is deliberate: this is a data-prep step, not a quality claim. If the
/// loss turns out sensitive to it, an anti-aliased resampler replaces it here.
pub fn resample_linear(x: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if x.is_empty() || from_hz == 0 || to_hz == 0 {
        return Vec::new();
    }
    if from_hz == to_hz {
        return x.to_vec();
    }

    let n_in = x.len();
    // round-half-up without float drift: (n_in · to + from/2) / from
    let n_out =
        ((n_in as u128 * to_hz as u128 + from_hz as u128 / 2) / from_hz as u128) as usize;

    (0..n_out)
        .map(|j| {
            let pos = j as f64 * from_hz as f64 / to_hz as f64;
            let i0 = pos.floor() as usize;
            let frac = (pos - i0 as f64) as f32;
            let a = x[i0.min(n_in - 1)];
            let b = x[(i0 + 1).min(n_in - 1)];
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_doubling_matches_repeat_interleave() {
        assert_eq!(nearest_index_map(4, 8), vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn empty_when_either_side_is_zero() {
        assert!(nearest_index_map(0, 10).is_empty());
        assert!(nearest_index_map(10, 0).is_empty());
        assert!(nearest_index_map(0, 0).is_empty());
    }

    #[test]
    fn single_source_frame_maps_everything_to_zero() {
        assert_eq!(nearest_index_map(1, 5), vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn no_source_frame_is_skipped_when_upsampling() {
        let map = nearest_index_map(50, 86);
        assert_eq!(map.len(), 86);
        assert_eq!(map[0], 0);
        assert_eq!(*map.last().unwrap(), 49);

        // every source index appears, and each step is 0 or 1
        let mut seen = [false; 50];
        for &i in &map {
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "some source frame was skipped");
        assert!(map.windows(2).all(|w| w[1] - w[0] <= 1));
    }

    #[test]
    fn real_ratios() {
        // 6.84 s: HuBERT 50 Hz → 342 frames, latent 86.13 Hz → 589 frames.
        let map = nearest_index_map(342, 589);
        assert_eq!(map.len(), 589);
        assert_eq!(map[0], 0);
        assert_eq!(*map.last().unwrap(), 341);
        assert!(map.iter().all(|&i| i < 342));

        // 600 s: 30000 → 51680.
        let map = nearest_index_map(30000, 51680);
        assert_eq!(map.len(), 51680);
        assert_eq!(*map.last().unwrap(), 29999);
    }

    #[test]
    fn non_decreasing_in_general() {
        for &(t_in, t_out) in &[(3, 7), (7, 3), (100, 137), (137, 100), (1, 1)] {
            let map = nearest_index_map(t_in, t_out);
            assert_eq!(map.len(), t_out);
            assert!(map.windows(2).all(|w| w[0] <= w[1]), "{t_in}->{t_out}");
            assert!(map.iter().all(|&i| i < t_in), "{t_in}->{t_out}");
        }
    }

    #[test]
    fn downsampling_picks_evenly_spaced_sources() {
        assert_eq!(
            nearest_index_map(4, 2),
            vec![0, 2],
            "floor(t·4/2) = 2t"
        );
    }

    #[test]
    fn identity() {
        assert_eq!(nearest_index_map(5, 5), vec![0, 1, 2, 3, 4]);
    }

    // ------------------------------------------------ sample-rate conversion

    #[test]
    fn resample_identity_when_rates_equal() {
        let x = [0.1, -0.2, 0.3, 0.4];
        assert_eq!(resample_linear(&x, 16000, 16000), x.to_vec());
    }

    #[test]
    fn resample_empty_and_zero_rate() {
        assert!(resample_linear(&[], 16000, 22050).is_empty());
        assert!(resample_linear(&[1.0, 2.0], 0, 22050).is_empty());
        assert!(resample_linear(&[1.0, 2.0], 16000, 0).is_empty());
    }

    #[test]
    fn resample_length_one_second_16k_to_22k() {
        let x = vec![0.0f32; 16000];
        assert_eq!(resample_linear(&x, 16000, 22050).len(), 22050);
        // and the real target wav length: 6.84 s = 109440 samples
        // 109440 · 22050 / 16000 = 150822 exactly
        let x = vec![0.0f32; 109440];
        assert_eq!(resample_linear(&x, 16000, 22050).len(), 150822);
    }

    #[test]
    fn resample_preserves_constant() {
        let x = vec![0.7f32; 1000];
        let y = resample_linear(&x, 16000, 22050);
        assert!(y.iter().all(|&v| (v - 0.7).abs() < 1e-6));
    }

    #[test]
    fn resample_single_sample() {
        assert_eq!(resample_linear(&[0.5], 16000, 22050), vec![0.5]);
    }

    #[test]
    fn resample_ramp_endpoints_and_monotone() {
        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y = resample_linear(&x, 100, 150);
        assert_eq!(y.len(), 150);
        assert!((y[0] - 0.0).abs() < 1e-4);
        assert!((y[y.len() - 1] - 99.0).abs() < 1e-3);
        assert!(y.windows(2).all(|w| w[0] <= w[1] + 1e-4));
    }
}
