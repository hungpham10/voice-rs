//! F0 decode: salience [360, T] -> f0 [T]

use burn::tensor::{Tensor, Float};

pub const N_CLASS: usize = 360;
pub const THRED: f32 = 0.03; // Applio's filter_radius, NOT 0.5 from paper

/// cents_mapping = 20*i + 1997.3794084376191, pad ±4 at both ends
fn cents_mapping() -> [f32; N_CLASS + 8] {
    let mut m = [0.0f32; N_CLASS + 8];
    for i in 0..N_CLASS {
        m[i + 4] = 20.0 * i as f32 + 1997.3794084376191;
    }
    m
}

/// Decode salience [360, T] to f0 [T] in Hz.
/// Applio's RMVPE decode logic:
/// - Find argmax per frame
/// - Weighted average ±4 bins around argmax using cents_mapping
/// - Threshold at 0.03 (NOT 0.5 from paper)
/// - f0 == 10.0 -> 0.0 fix
pub fn decode(salience: &Tensor<3, Float>) -> Vec<f32> {
    let shape = salience.shape();
    let n_frames = shape.dims::<3>()[2];
    let cm = cents_mapping();
    let mut out = vec![0.0f32; n_frames];

    // salience is [1, 360, T]
    let data = salience.to_data();
    let vals = data.as_slice::<f32>().expect("salience is f32");

    for t in 0..n_frames {
        // Find argmax for this frame
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for c in 0..N_CLASS {
            let idx = c * n_frames + t; // [1, 360, T] layout
            let v = vals[idx];
            if v > max_val {
                max_val = v;
                max_idx = c;
            }
        }

        // Threshold check
        if max_val <= THRED {
            out[t] = 0.0;
            continue;
        }

        // Weighted average ±4 bins
        // Python pads salience with zeros at both ends (pad_width=4), then adds 4 to center
        // So for c = m + k - 4:
        // - If c < 0 or c >= 360, the padded salience is 0 (weight = 0)
        // - cents_mapping is padded with 4 zeros on each side (cm[0..3] = cm[364..367] = 0)
        // We replicate this: use zero weight for out-of-bounds bins
        let m = max_idx as i32;
        let mut num = 0.0;
        let mut den = 0.0;
        for k in 0..9 {
            let c = m + k - 4;
            // Zero padding: if c is out of bounds, weight = 0
            let (w, cent) = if c < 0 || c >= N_CLASS as i32 {
                (0.0f32, 0.0f32)
            } else {
                let idx = c as usize * n_frames + t;
                (vals[idx], cm[c as usize + 4])
            };
            num += w * cent;
            den += w;
        }

        let cents = num / den;
        let f0 = 10.0 * 2f32.powf(cents / 1200.0);

        // Applio's fix: f0 == 10.0 -> 0.0
        out[t] = if f0 == 10.0 { 0.0 } else { f0 };
    }

    out
}

/// Quantize f0 to coarse pitch bins (for feature matching)
/// f0_bin = 256, f0_max = 1100.0, f0_min = 50.0
/// f0_mel = 1127 * ln(1 + f0/700)
/// f0_mel = clip((f0_mel - mel_min)*(f0_bin-2)/(mel_max-mel_min) + 1, 1, f0_bin-1)
/// f0_mel = round(f0_mel)
pub fn quantize(f0: &[f32]) -> Vec<i32> {
    const BIN: f32 = 256.0;
    const MIN: f32 = 50.0;
    const MAX: f32 = 1100.0;
    let mel = |f: f32| 1127.0 * (1.0 + f / 700.0).ln();
    let lo = mel(MIN);
    let hi = mel(MAX);
    let scale = (BIN - 2.0) / (hi - lo);

    f0.iter()
        .map(|&f| {
            let m = (mel(f.max(0.0)) - lo) * scale + 1.0;
            m.clamp(1.0, BIN - 1.0).round() as i32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cents_mapping() {
        let cm = cents_mapping();
        // First valid entry at index 4 (after 4 padding)
        assert!((cm[4] - 1997.3794084376191).abs() < 1e-6);
        // Last valid entry at index 363 (4 + 360 - 1)
        assert!((cm[363] - (20.0 * 359.0 + 1997.3794084376191)).abs() < 1e-6);
    }

    #[test]
    fn test_decode_peak_at_center() {
        // Create fake salience with peak at bin 100
        let n_frames = 10;
        let mut data = vec![0.0f32; N_CLASS * n_frames];
        for t in 0..n_frames {
            data[100 * n_frames + t] = 1.0;
        }
        let salience = Tensor::<3, Float>::from_data(
            TensorData::new(data, [1, N_CLASS, n_frames]), 
            &Default::default()
        );

        let f0 = decode(&salience);
        // All frames should have same f0
        for &v in &f0 {
            assert!(v > 0.0);
        }
        // Check value: cents = 20*100 + 1997.38 = 3997.38
        // f0 = 10 * 2^(3997.38/1200) = 10 * 2^3.33115 = 10 * 10.056 = ~100.56 Hz
        let expected = 10.0 * 2f32.powf((20.0 * 100.0 + 1997.3794084376191) / 1200.0);
        assert!((f0[0] - expected).abs() < 0.1);
    }

    #[test]
    fn test_quantize() {
        let f0 = vec![50.0, 100.0, 200.0, 400.0, 800.0, 1100.0, 0.0];
        let q = quantize(&f0);
        // 50 Hz -> bin 1
        assert_eq!(q[0], 1);
        // 1100 Hz -> bin 255
        assert_eq!(q[5], 255);
        // 0 Hz -> mel(0) = 0 -> clamp to 1
        assert_eq!(q[6], 1);
    }

    /// Regression test: decode should use zero padding for out-of-bounds bins
    /// (matching Python: np.pad(salience, ((0,0),(4,4))) then center += 4)
    #[test]
    fn test_decode_zero_padding() {
        // Peak at bin 0 (should use zero padding for left side)
        let n_frames = 5;
        let mut data = vec![0.0f32; N_CLASS * n_frames];
        for t in 0..n_frames {
            data[0 * n_frames + t] = 1.0;
            // Small values at bins 1-4 so argmax stays at 0
            for c in 1..5 {
                data[c * n_frames + t] = 0.1;
            }
        }
        let salience = Tensor::<3, Float>::from_data(
            TensorData::new(data, [1, N_CLASS, n_frames]),
            &Default::default()
        );

        let f0 = decode(&salience);
        // All frames should have same f0 (bin 0 -> ~32.7 Hz but thresholded if low)
        for &v in &f0 {
            // With zero padding, the weighted average will be pulled toward
            // the valid bins. Verify it doesn't panic and produces valid output.
            assert!(v.is_finite());
        }
    }

    /// Peak at bin 359 (should use zero padding for right side)
    #[test]
    fn test_decode_zero_padding_right() {
        let n_frames = 5;
        let mut data = vec![0.0f32; N_CLASS * n_frames];
        for t in 0..n_frames {
            data[359 * n_frames + t] = 1.0;
            for c in 355..359 {
                data[c * n_frames + t] = 0.1;
            }
        }
        let salience = Tensor::<3, Float>::from_data(
            TensorData::new(data, [1, N_CLASS, n_frames]),
            &Default::default()
        );

        let f0 = decode(&salience);
        for &v in &f0 {
            assert!(v.is_finite());
        }
    }
}