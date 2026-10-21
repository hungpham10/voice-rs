//! Log-mel **khả vi** bằng Burn + loss `L1`.
//!
//! Không dùng `voice_mel::MelConfig::spectrogram` cho đường huấn luyện: nó dựa
//! trên `rosa` (CPU, không có gradient). Ở đây STFT/Hann/mel-filter được dựng lại
//! bằng op của Burn (`burn::signal::stft`) để gradient chảy ngược qua `dec` về
//! `Adapter`.
//!
//! [`log_mel`] được viết để **khớp** `MelConfig::spectrogram`: biên độ (không
//! phải công suất), áp `mel_floor` **trước** log, cùng cửa sổ Hann tuần hoàn và
//! cùng cách đệm reflect `center = true`. Test `burn_mel_matches_rosa` giữ hai
//! bên khớp nhau.

use burn::signal::{hann_window, stft, StftOptions};
use burn::tensor::{Device, Tensor, TensorData};
use voice_mel::{LogMel, MelConfig};

/// Ma trận mel `[n_freq, n_mels]` dưới dạng tensor Burn (F32) trên `device`.
///
/// Lấy thẳng filterbank đã kiểm của `voice-mel` để hai đường không lệch thang.
pub fn mel_basis(cfg: &MelConfig, device: &Device) -> Tensor<2> {
    let fb = cfg.filterbank(); // [n_freq][n_mels]
    let n_freq = fb.len();
    let n_mels = fb[0].len();
    let mut flat = Vec::with_capacity(n_freq * n_mels);
    for row in &fb {
        for &v in row {
            flat.push(v as f32);
        }
    }
    Tensor::<2>::from_data(TensorData::new(flat, [n_freq, n_mels]), device)
}

/// Log-mel khả vi: `wav [B, N]` → `[B, n_frames, n_mels]`.
pub fn log_mel(wav: Tensor<2>, cfg: &MelConfig, basis: &Tensor<2>) -> Tensor<3> {
    let opts = StftOptions {
        n_fft: cfg.n_fft,
        hop_length: cfg.hop_length,
        win_length: Some(cfg.n_fft),
        center: cfg.center,
        onesided: true,
    };
    let window = hann_window(cfg.n_fft, true, &wav.device());

    // [B, n_frames, n_freq, 2]
    let spec = stft(wav, Some(window), opts);

    let re = spec.clone().narrow(3, 0, 1).squeeze_dim::<3>(3);
    let im = spec.narrow(3, 1, 1).squeeze_dim::<3>(3);
    let mag = (re.powf_scalar(2.0) + im.powf_scalar(2.0)).sqrt(); // [B, F, n_freq]

    let [b, f, n_freq] = mag.dims();
    let mel = mag
        .reshape([b * f, n_freq])
        .matmul(basis.clone())
        .reshape([b, f, cfg.n_mels]);

    let mel = mel.clamp_min(cfg.mel_floor);
    match cfg.log_mel {
        LogMel::None => mel,
        LogMel::Ln => mel.log(),
        LogMel::Log10 => mel.log().div_scalar(core::f32::consts::LN_10),
    }
}

/// `L1` giữa log-mel của `gen` và của `target` (cả hai `[B, N]`), lấy trung bình.
///
/// Trả về `Tensor<1>` để gọi `backward()` trực tiếp (burn gieo seed 1 cho mọi
/// phần tử, tensor một phần tử nên tương đương scalar).
pub fn mel_l1(gen: Tensor<2>, target: Tensor<2>, cfg: &MelConfig, basis: &Tensor<2>) -> Tensor<1> {
    let a = log_mel(gen, cfg, basis);
    let b = log_mel(target, cfg, basis);
    (a - b).abs().mean()
}

/// `mel_l1` có mặt nạ frame: `mask [B, F]` (1 = frame mel thật, 0 = vùng pad).
///
/// Batch pad 0 về `chunk_frames` để shape đều nhau; vùng pad qua gen sinh nhiễu
/// mép STFT — mask loại khỏi loss: `Σ|a−b|·mask / Σmask`.
/// Frame thật: `t·hop < real_samples` (dựng ở `train.rs` từ độ dài chunk nguyên bản).
pub fn mel_l1_masked(
    gen: Tensor<2>,
    target: Tensor<2>,
    cfg: &MelConfig,
    basis: &Tensor<2>,
    mask: &Tensor<2>,
) -> Tensor<1> {
    let a = log_mel(gen, cfg, basis);
    let b = log_mel(target, cfg, basis);
    // mask [B, F] → [B, F, 1] broadcast nhân vào mel [B, F, n_mels].
    let [mb, mf] = mask.dims();
    let m = mask.clone().reshape([mb, mf, 1]);
    // KHUYẾN CẢO: den phải chia cho TẤT CẢ phần tử mà num đã cộng (B×F×n_mels),
    // không chỉ số frame (B×F) — nếu không loss sẽ bị phóng đại ~80×
    // (số n_mels) và học lên NaN (khác ma trận Applio: họ crop đều → không mask,
    // vậy không gặp bẫy divisor này).
    let num = (a - b).abs().mul(m).sum();
    let n_mels = cfg.n_mels as f32;
    let den = mask.clone().sum() * n_mels;
    num.div(den)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(n: usize, sr: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                (2.0 * core::f32::consts::PI * 220.0 * t).sin() * 0.5
                    + (2.0 * core::f32::consts::PI * 440.0 * t).sin() * 0.3
            })
            .collect()
    }

    /// Đường burn phải khớp đường `rosa` tham chiếu — nếu lệch, loss sẽ tối ưu
    /// sai thang.
    #[test]
    fn burn_mel_matches_rosa() {
        let device = Device::flex();
        let cfg = MelConfig::ljs();
        let n = 22050; // 1 s @ 22.05 kHz
        let x = sine(n, 22050.0);

        let basis = mel_basis(&cfg, &device);
        let wav = Tensor::<2>::from_data(TensorData::new(x.clone(), [1, n]), &device);
        let got = log_mel(wav, &cfg, &basis).into_data().as_slice::<f32>().unwrap().to_vec();
        let [_, f, m] = log_mel(
            Tensor::<2>::from_data(TensorData::new(x.clone(), [1, n]), &device),
            &cfg,
            &basis,
        )
        .dims();
        assert_eq!(m, 80);

        let reference = cfg.spectrogram(&x).unwrap(); // [80, frames]
        assert_eq!(reference[0].len(), f, "so frame lech");

        let mut worst = 0.0f32;
        let mut mean = 0.0f32;
        for frame in 0..f {
            for bin in 0..m {
                let g = got[frame * m + bin];
                let e = reference[bin][frame];
                let d = (g - e).abs();
                worst = worst.max(d);
                mean += d;
            }
        }
        mean /= (f * m) as f32;
        println!("burn vs rosa mel: max={worst:.3e} mean={mean:.3e} ({f} frame)");
        assert!(worst < 1e-2, "mel burn lech qua lon: max={worst}");
    }

    #[test]
    fn mel_l1_zero_for_identical_audio() {
        let device = Device::flex();
        let cfg = MelConfig::ljs();
        let basis = mel_basis(&cfg, &device);
        let x = sine(8192, 22050.0);
        let a = Tensor::<2>::from_data(TensorData::new(x.clone(), [1, 8192]), &device);
        let b = Tensor::<2>::from_data(TensorData::new(x, [1, 8192]), &device);
        let l: f32 = mel_l1(a, b, &cfg, &basis).into_scalar();
        assert!(l.abs() < 1e-4, "loss phai ~0, got {l}");
    }

    /// Masked loss phải bỏ đúng vùng pad: phần gen bị thay 0 (nhiễu mép) vẫn
    /// không ảnh hưởng loss nếu nằm ngoài mask.
    #[test]
    fn mel_l1_masked_ignores_padding() {
        let device = Device::flex();
        let cfg = MelConfig::ljs();
        let basis = mel_basis(&cfg, &device);

        let real = 4096usize; // samples thật
        let n = 8192usize; // tổng (đã pad 0)
        let pad_len = n - real;
        let x = sine(real, 22050.0);
        // gen: audio thật + phần pad là nhiễu lớn; target: audio thật + pad 0
        let mut gen_wav = x.clone();
        gen_wav.extend(std::iter::repeat(0.5f32).take(pad_len));
        let mut tgt_wav = x.clone();
        tgt_wav.extend(std::iter::repeat(0.0f32).take(pad_len));

        let hop = cfg.hop_length as usize;
        let probe = log_mel(
            Tensor::<2>::from_data(TensorData::new(gen_wav.clone(), [1, n]), &device),
            &cfg,
            &basis,
        );
        let [_, f, _] = probe.dims();
        // mask: 1 cho frame t khi window [t*hop − n_fft/2, t*hop + n_fft/2)
        // không đụng vùng pad → t*hop + n_fft/2 <= real
        let mut mask = vec![0f32; f];
        let n_fft_half = cfg.n_fft / 2;
        for t in 0..f {
            if t * hop + n_fft_half <= real {
                mask[t] = 1.0;
            }
        }
        let mask_t = Tensor::<2>::from_data(TensorData::new(mask, [1, f]), &device);
        let gen_t = Tensor::<2>::from_data(TensorData::new(gen_wav, [1, n]), &device);
        let tgt_t = Tensor::<2>::from_data(TensorData::new(tgt_wav, [1, n]), &device);
        let l: f32 = mel_l1_masked(gen_t.clone(), tgt_t.clone(), &cfg, &basis, &mask_t).into_scalar();
        assert!(l.abs() < 1e-3, "masked loss phải bỏ qua vùng pad, got {l}");

        // đối chứng: không mask thì khác 0 rõ
        let l_all: f32 = mel_l1(gen_t, tgt_t, &cfg, &basis).into_scalar();
        assert!(l_all > 0.1, "không mask phải thấy phần pad, got {l_all}");
    }

    /// Full mask (mọi frame = 1) → masked loss phải bằng chính xác loss không mask,
    //  vì den giờ có nhân n_mels (không còn sai ~80× gây NaN).
    #[test]
    fn mel_l1_masked_full_mask_equals_plain() {
        let device = Device::flex();
        let cfg = MelConfig::ljs();
        let basis = mel_basis(&cfg, &device);

        let x = sine(8192, 22050.0);
        let y = (0..8192usize).map(|i| {
            let t = i as f32 / 22050.0;
            (2.0 * core::f32::consts::PI * 330.0 * t).sin() * 0.4
        }).collect::<Vec<_>>();
        let x_t = Tensor::<2>::from_data(TensorData::new(x, [1, 8192]), &device);
        let y_t = Tensor::<2>::from_data(TensorData::new(y, [1, 8192]), &device);

        // full mask đủ số frame mel của audio 8192 mẫu
        let probe = log_mel(x_t.clone(), &cfg, &basis);
        let [_, f, _] = probe.dims();
        let mask = Tensor::<2>::from_data(TensorData::new(vec![1.0f32; f], [1, f]), &device);

        let l_masked: f32 = mel_l1_masked(x_t.clone(), y_t.clone(), &cfg, &basis, &mask).into_scalar();
        let l_plain: f32 = mel_l1(x_t, y_t, &cfg, &basis).into_scalar();
        assert!((l_masked - l_plain).abs() < 1e-5,
                "full mask phải bằng plain: masked={l_masked} plain={l_plain}");
        assert!(l_masked > 0.1, "âm khác nhau phải có loss rõ ràng, got {l_masked}");
    }

    /// Gradient qua STFT/`log` phải finite (chặn bẫy NaN §8.2) và khác 0.
    #[test]
    fn mel_l1_gradient_is_finite_and_nonzero() {
        let device = Device::flex().autodiff();
        let cfg = MelConfig::ljs();
        let basis = mel_basis(&cfg, &device);

        let x = sine(8192, 22050.0);
        let gen = Tensor::<2>::from_data(TensorData::new(x.clone(), [1, 8192]), &device)
            .require_grad();
        let tgt = Tensor::<2>::from_data(TensorData::new(vec![0.0f32; 8192], [1, 8192]), &device);
        let loss = mel_l1(gen.clone(), tgt, &cfg, &basis);
        let grads = loss.backward();
        let g = gen.grad(&grads).expect("phai co gradient");
        let vals = g.into_data().as_slice::<f32>().unwrap().to_vec();
        assert!(vals.iter().all(|v| v.is_finite()), "gradient NaN/Inf");
        assert!(vals.iter().any(|v| v.abs() > 1e-9), "gradient bang 0");
    }
}
