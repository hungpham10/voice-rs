//! SineGen — sinh **excitation** từ f0, **0 tham số** (§13.4 NSF-lite).
//!
//! `f0 [B, T_z]` (frame-rate `LATENT_HZ = 86.13 Hz`) → excitation `[B, K, T_z]`
//! với `K = 8` thành phần sin. Thành phần thứ `k` (0..K):
//!   φ[k,t] = 2π · Σ_{s≤t} (k+1)·f0[s] / LATENT_HZ          (phase cumsum)
//!   exc[k,t] = sin(φ[k,t]) · 1[f0[t] > 0]                  (voiced mask)
//!
//! Không thêm Gaussian noise (khác NSF gốc): conv_src học từ đầu sẽ lấp phần
//! còn lại. Tất cả bằng op Burn, hoàn toàn khả vi gradient.
//!
//! Vị trí bơm: **sau `conv_pre`** — excitation ở cùng frame-rate `T_z` với `x`
//! sau `conv_pre` (`[B, 512, T_z]`), nên `conv_src` Conv1d(8→512, k=1) cộng
//! trực tiếp vào `x` trước chu kỳ upsample.

use burn::tensor::{Device, Tensor, TensorData};

/// Số thành phần harmonics của SineGen (gốc + 7 harmonic cao hơn).
pub const SINE_HARMONICS: usize = 8;

/// Tần số frame latent của `dec` (`22050 / 256 = 86.1328125 Hz`).
pub const SINE_RATE: f32 = 86.1328125;

/// SineGen thuần toán, không tham số huấn luyện.
#[derive(Debug, Clone, Copy, Default)]
pub struct SineGen;

impl SineGen {
    /// `f0 [B, T_z]` (Hz) → excitation `[B, K, T_z]`.
    ///
    /// - `f0 == 0` (unvoiced) → toàn bộ excitation channels = 0.
    /// - Mỗi channel `k` tần số `(k+1)·f0`, phase tích lũy qua các frame.
    pub fn forward(f0: Tensor<2>, device: &Device) -> Tensor<3> {
        let b = f0.dims()[0];
        let t = f0.dims()[1];

        // Đọc dữ liệu f0 một lần (consume tensor) → dùng cho cả phase và voiced mask.
        let f0_data = f0.into_data();
        let f0_slice = f0_data.as_slice::<f32>().unwrap();
        let f0_vals: Vec<f32> = f0_slice.to_vec();
        let mask_vals: Vec<f32> = f0_slice
            .iter()
            .map(|&v| if v > 0.0 { 1.0f32 } else { 0.0f32 })
            .collect();

        // y[t] = f0[t] / rate   (cycles per frame)  [B, T_z]
        let y = Tensor::<2>::from_data(TensorData::new(f0_vals, [b, t]), device)
            .div_scalar(SINE_RATE);

        // scalars [1, K, 1]: (k+1) cho mỗi harmonic
        let scalars: Vec<f32> = (1..=SINE_HARMONICS).map(|k| k as f32).collect();
        let harmonics = Tensor::<3>::from_data(
            TensorData::new(scalars, [1, SINE_HARMONICS, 1]),
            device,
        );

        // freqs[k,t] = (k+1)·y[t]   [B, K, T_z]
        let freqs = y.reshape([b, 1, t]) * harmonics;

        // phase = 2π · cumsum(freqs, dim=T_z)   [B, K, T_z]
        let phase = freqs.cumsum(2) * (2.0 * std::f32::consts::PI);

        // exc = sin(phase) × voiced_mask   [B, K, T_z]
        let exc = phase.sin();
        let mask = Tensor::<2>::from_data(TensorData::new(mask_vals, [b, t]), device)
            .reshape([b, 1, t]);
        let exc = exc * mask;

        exc
    }

    /// Số channels excitation đầu ra (giống `SINE_HARMONICS`).
    pub fn out_channels() -> usize {
        SINE_HARMONICS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Device {
        Device::flex()
    }

    #[test]
    fn sinegen_shape_and_dtype() {
        let device = device();
        let f0 = Tensor::<2>::from_data(
            TensorData::new(vec![220.0f32, 220.0, 220.0], [1, 3]),
            &device,
        );
        let exc = SineGen::forward(f0, &device);
        let d = exc.dims();
        assert_eq!(d[0], 1);
        assert_eq!(d[1], SINE_HARMONICS);
        assert_eq!(d[2], 3);
    }

    #[test]
    fn sinegen_unvoiced_is_zero() {
        let device = device();
        let f0 = Tensor::<2>::from_data(
            TensorData::from([[0.0f32, 0.0, 0.0, 0.0]]),
            &device,
        );
        let exc = SineGen::forward(f0, &device);
        let data = exc.into_data().as_slice::<f32>().unwrap().to_vec();
        assert!(data.iter().all(|&v| v == 0.0), "unvoiced phải toàn 0");
    }

    #[test]
    fn sinegen_golden_phase() {
        let device = device();
        // f0 hằng 440 Hz @ 86.13 Hz: frame 0 → phase 0, frame 1 → 2π·440/86.13, …
        let f0 = Tensor::<2>::from_data(
            TensorData::new(vec![440.0f32, 440.0, 440.0], [1, 3]),
            &device,
        );
        let exc = SineGen::forward(f0, &device);
        let data = exc.into_data().as_slice::<f32>().unwrap().to_vec();
        let rate = SINE_RATE;
        for k in 0..SINE_HARMONICS {
            for t in 0..3usize {
                let expected = ((2.0 * std::f32::consts::PI)
                    * (k + 1) as f32
                    * 440.0f32
                    * ((t + 1) as f32 / rate))
                .sin();
                let got = data[k * 3 + t];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "harmonic {k} frame {t}: expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn sinegen_mixed_voiced_unvoiced() {
        let device = device();
        let f0 = Tensor::<2>::from_data(
            TensorData::new(vec![220.0f32, 0.0, 440.0], [1, 3]),
            &device,
        );
        let exc = SineGen::forward(f0, &device);
        let data = exc.into_data().as_slice::<f32>().unwrap().to_vec();
        // Frame 1 (unvoiced) phải toàn 0
        for k in 0..SINE_HARMONICS {
            assert_eq!(data[k * 3 + 1], 0.0, "frame 1 unvoiced phải 0");
        }
        // Frame 0 và 2 phải khác 0 (ít nhất 1 channel)
        for t in [0usize, 2usize] {
            let nonzero = (0..SINE_HARMONICS).any(|k| data[k * 3 + t] != 0.0);
            assert!(nonzero, "frame {t} voiced phải có excitation khác 0");
        }
    }
}
