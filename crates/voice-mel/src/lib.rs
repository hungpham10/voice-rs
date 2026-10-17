//! voice-mel — Mel spectrogram extraction matching librosa/RMVPE conventions.
//!
//! Provides two pre-configured mel types:
//! - `MelConfig::rmvpe()` — HTK mel scale, fmin=30, fmax=8000, center=true
//!   (matches Applio's RMVPE frontend exactly)
//! - `MelConfig::rvc()` — Slaney mel scale, fmin=0, fmax=8000, custom padding
//!   (for RVC synthesizer input, used later)

use rosa::stft::{
    mel_filter, StftParams, MelFilterParams, MelNorm,
    stft_as_real_imag,
};
use rosa::Matrix;
use thiserror::Error;

/// Logarithm applied to the mel values.
///
/// RMVPE/RVC use natural log; SpeechT5 uses log10. SpeechT5 also floors at
/// `1e-10` (vs `1e-5` for RMVPE), so the floor is configurable too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogMel {
    /// Linear mel, no logarithm.
    None,
    /// Natural log.
    Ln,
    /// Base-10 log.
    Log10,
}

/// Configuration for mel spectrogram extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct MelConfig {
    pub n_mels: usize,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub htk: bool,
    pub norm: MelNorm,
    pub log_mel: LogMel,
    pub mel_floor: f32,
    pub center: bool,
}

impl MelConfig {
    /// Configuration matching Applio's RMVPE frontend exactly.
    /// Uses HTK mel scale, fmin=30, fmax=8000, center=true, natural log.
    pub fn rmvpe() -> Self {
        Self {
            n_mels: 128,
            sample_rate: 16000,
            n_fft: 1024,
            hop_length: 160,
            fmin: 30.0,
            fmax: 8000.0,
            htk: true,
            norm: MelNorm::None,
            log_mel: LogMel::Ln,
            mel_floor: 1e-5,
            center: true,
        }
    }

    /// Configuration for RVC synthesizer input (Slaney mel).
    /// Will be used later when implementing the full pipeline.
    pub fn rvc() -> Self {
        Self {
            n_mels: 128,
            sample_rate: 16000,
            n_fft: 1024,
            hop_length: 160,
            fmin: 0.0,
            fmax: 8000.0,
            htk: false,
            norm: MelNorm::Slaney,
            log_mel: LogMel::Ln,
            mel_floor: 1e-5,
            center: false, // RVC uses custom padding, not center=True
        }
    }

    /// Configuration matching HF `SpeechT5FeatureExtractor` (used by the
    /// `microsoft/speecht5_hifigan` vocoder).
    ///
    /// - 16 kHz, `n_fft = 1024` (win 64 ms), `hop = 256` (16 ms), 80 mel bins
    /// - `fmin = 80`, `fmax = 7600`
    /// - Slaney mel scale + Slaney filterbank normalization
    /// - amplitude spectrogram (`power = 1.0`)
    /// - `center = true` (reflect padding), so frames `= 1 + len / hop`
    /// - mel floored at `1e-10`, then **log10**
    pub fn speecht5() -> Self {
        Self {
            n_mels: 80,
            sample_rate: 16000,
            n_fft: 1024,
            hop_length: 256,
            fmin: 80.0,
            fmax: 7600.0,
            htk: false,
            norm: MelNorm::Slaney,
            log_mel: LogMel::Log10,
            mel_floor: 1e-10,
            center: true,
        }
    }

    /// Configuration for the VITS `dec` training loss (22.05 kHz).
    ///
    /// Matches the generator output rate: `n_fft = 1024`, `hop = 256` (16 ms),
    /// 80 mel bins, `fmin = 0`, `fmax = 8000`, Slaney mel + Slaney norm,
    /// amplitude magnitude, floored at `1e-5` then natural log. `center = true`.
    ///
    /// Only the *loss* uses this; the source audio is 16 kHz and is resampled up
    /// with `voice_lib::vc::resample_linear` before comparison.
    pub fn ljs() -> Self {
        Self {
            n_mels: 80,
            sample_rate: 22050,
            n_fft: 1024,
            hop_length: 256,
            fmin: 0.0,
            fmax: 8000.0,
            htk: false,
            norm: MelNorm::Slaney,
            log_mel: LogMel::Ln,
            mel_floor: 1e-5,
            center: true,
        }
    }

    /// Number of frames produced for a waveform of `len` samples.
    ///
    /// With `center = true` the waveform is reflect-padded by `n_fft / 2` on
    /// both sides (librosa convention), giving `1 + len / hop`.
    pub fn num_frames(&self, len: usize) -> usize {
        if self.center {
            1 + len / self.hop_length
        } else if len <= self.n_fft {
            0
        } else {
            1 + (len - self.n_fft) / self.hop_length
        }
    }

    /// Build the mel filterbank matrix [n_freq, n_mels] where n_freq = n_fft/2 + 1.
    /// Returns row-major matrix for easier matmul: magnitude @ mel_basis
    pub fn filterbank(&self) -> Vec<Vec<f64>> {
        let mel_params = MelFilterParams {
            sr: self.sample_rate as f64,
            n_fft: self.n_fft,
            n_mels: self.n_mels,
            fmin: self.fmin as f64,
            fmax: Some(self.fmax as f64),
            htk: self.htk,
            norm: self.norm.clone(),
        };
        let filter: Matrix = mel_filter(&mel_params); // [n_mels, n_freq]
        // Transpose to [n_freq, n_mels]
        let n_mels = filter.rows();
        let n_freq = filter.cols();
        let mut result = vec![vec![0.0f64; n_mels]; n_freq];
        for i in 0..n_freq {
            for m in 0..n_mels {
                result[i][m] = filter.get(m, i);
            }
        }
        result
    }

    /// Compute mel spectrogram from waveform.
    /// Returns log-mel spectrogram [n_mels, n_frames] with amplitude magnitude.
    pub fn spectrogram(&self, wav: &[f32]) -> Result<Vec<Vec<f32>>, MelError> {
        // Convert f32 to f64 for rosa
        let wav_f64: Vec<f64> = wav.iter().map(|&x| x as f64).collect();

        // Match torch.stft(center=True) which uses reflect padding.
        // rosa's center=True uses zero-padding (librosa convention), so we
        // manually apply reflect padding and call rosa with center=false.
        let wav_padded_f64: Vec<f64>;
        let center = self.center;
        let wav_for_stft = if center {
            let n_half = self.n_fft / 2;
            if wav_f64.len() <= n_half {
                return Err(MelError::InvalidInput(format!(
                    "Input length {} must be > n_fft/2 ({}) for reflect padding",
                    wav_f64.len(),
                    n_half
                )));
            }
            // Reflect pad: np.pad(mode='reflect') matches torch.stft default
            // Example: [a,b,c,d] with pad=2 -> [c,b,a,b,c,d,c,b]
            let mut padded = Vec::with_capacity(wav_f64.len() + 2 * n_half);
            // Left pad (reflect, excluding edge)
            for i in (1..=n_half).rev() {
                padded.push(wav_f64[i.min(wav_f64.len() - 1)]);
            }
            // Original signal
            padded.extend_from_slice(&wav_f64);
            // Right pad (reflect, excluding edge)
            for i in 1..=n_half {
                let idx = (wav_f64.len() - 1).saturating_sub(i);
                padded.push(wav_f64[idx]);
            }
            wav_padded_f64 = padded;
            &wav_padded_f64
        } else {
            &wav_f64
        };

        // STFT parameters
        let stft_params = StftParams {
            n_fft: self.n_fft,
            hop_length: self.hop_length,
            win_length: self.n_fft, // win_length = n_fft per rmvpe.py
            center: false, // We handle padding manually
            window: None, // default periodic hann
        };

        // Compute STFT - returns (re, im) as Matrix [n_freq, n_frames]
        let (re, im) = stft_as_real_imag(wav_for_stft, &stft_params);

        let n_freq = re.rows();
        let n_frames = re.cols();

        // Convert to amplitude magnitude: sqrt(re^2 + im^2)
        let mut magnitude = vec![vec![0.0f32; n_frames]; n_freq];
        for i in 0..n_freq {
            for j in 0..n_frames {
                let r = re.get(i, j) as f32;
                let im = im.get(i, j) as f32;
                magnitude[i][j] = (r * r + im * im).sqrt();
            }
        }

        // Apply mel filterbank: magnitude^T @ mel_basis -> [n_frames, n_mels]
        let mel_basis = self.filterbank(); // [n_freq, n_mels]
        let n_mels = self.n_mels;

        // matmul: [n_frames, n_mels] = [n_frames, n_freq] @ [n_freq, n_mels]
        let mut mel = vec![vec![0.0f32; n_mels]; n_frames];
        for j in 0..n_frames {
            for m in 0..n_mels {
                let mut sum = 0.0f32;
                for i in 0..n_freq {
                    sum += magnitude[i][j] * mel_basis[i][m] as f32;
                }
                mel[j][m] = sum;
            }
        }

        // Transpose to [n_mels, n_frames] and apply max(mel_floor, ·) then the
        // configured logarithm (HF applies the floor *before* the log).
        let mut out = vec![vec![0.0f32; n_frames]; n_mels];
        for m in 0..n_mels {
            for j in 0..n_frames {
                let v = mel[j][m].max(self.mel_floor);
                out[m][j] = match self.log_mel {
                    LogMel::None => v,
                    LogMel::Ln => v.ln(),
                    LogMel::Log10 => v.log10(),
                };
            }
        }

        Ok(out)
    }
}

#[derive(Debug, Error)]
pub enum MelError {
    #[error("Rosa STFT error: {0}")]
    Rosa(#[from] rosa::audio::AudioError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_wav_mono(path: &Path) -> (Vec<f32>, u32) {
        let mut reader = hound::WavReader::open(path).expect("Failed to open WAV");
        let spec = reader.spec();
        let samples: Vec<f32> = match (spec.bits_per_sample, spec.sample_format) {
            (16, hound::SampleFormat::Int) => reader
                .samples::<i16>()
                .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
                .collect(),
            (32, hound::SampleFormat::Int) => reader
                .samples::<i32>()
                .map(|s| s.expect("sample") as f32 / i32::MAX as f32)
                .collect(),
            (32, hound::SampleFormat::Float) => reader
                .samples::<f32>()
                .map(|s| s.expect("sample"))
                .collect(),
            _ => panic!("Unsupported WAV format: {} bits {:?}", spec.bits_per_sample, spec.sample_format),
        };
        (samples, spec.sample_rate)
    }

    #[test]
    fn test_hz_to_mel_htk() {
        // HTK formula: 2595 * log10(1 + f/700)
        let f = 1000.0;
        let mel = rosa::stft::hz_to_mel(f, true);
        let expected = 2595.0 * (1.0 + f / 700.0).log10();
        assert!((mel - expected).abs() < 1e-6);
    }

    #[test]
    fn test_hz_to_mel_slaney() {
        // Slaney formula from librosa - rosa uses its own implementation
        let f = 1000.0;
        let mel = rosa::stft::hz_to_mel(f, false);
        // Just verify it's a reasonable value
        assert!(mel > 0.0 && mel < 2000.0);
    }

    #[test]
    fn test_rmvpe_filterbank_shape() {
        let config = MelConfig::rmvpe();
        let fb = config.filterbank();
        assert_eq!(fb.len(), 513); // n_fft/2 + 1 = 1024/2 + 1
        assert_eq!(fb[0].len(), 128); // n_mels
    }

    #[test]
    fn test_rvc_filterbank_shape() {
        let config = MelConfig::rvc();
        let fb = config.filterbank();
        assert_eq!(fb.len(), 513);
        assert_eq!(fb[0].len(), 128);
    }

    #[test]
    fn test_filterbank_triangle_properties() {
        let config = MelConfig::rmvpe();
        let fb = config.filterbank();
        // Each column (mel bin) should have non-zero values forming a triangle shape
        for m in 0..128 {
            let mut max_val = 0.0f64;
            for i in 0..513 {
                max_val = max_val.max(fb[i][m]);
            }
            // Peak should be >0 for all bins (fmin=30 Hz means low bins are truncated)
            assert!(max_val > 0.1, "Mel bin {} max={}", m, max_val);
        }
    }

    #[test]
    fn test_spectrogram_shape() {
        let config = MelConfig::rmvpe();
        // 1 second of audio at 16kHz
        let wav = vec![0.1f32; 16000];
        let mel = config.spectrogram(&wav).unwrap();
        // n_frames = floor((16000 - 1024) / 160) + 1 = 94 + 1 = 95 with center=True
        // Actually with center=True: n_frames = 1 + len // hop = 1 + 16000 // 160 = 101
        assert_eq!(mel.len(), 128); // n_mels
        assert!(mel[0].len() >= 100 && mel[0].len() <= 102);
    }

    /// Regression test: verify mel padding matches torch.stft(center=True) with reflect mode.
    /// Values computed from Python torch.stft on reference.wav (first 10 mels of frame 0,1,2).
    #[test]
    fn test_rmvpe_mel_padding_reflect() {
        // Load reference.wav and compute mel with our implementation
        let wav_path = Path::new("../../../sources/Applio/logs/reference/reference.wav");
        if !wav_path.exists() {
            // Skip if test data not available
            return;
        }
        let samples = load_wav_mono(wav_path).0;
        let config = MelConfig::rmvpe();
        let mel = config.spectrogram(&samples).unwrap();

        // Expected values from torch.stft (librosa mel htk=True, reflect padding)
        // Frame 0 (first 10 mels)
        let expected_f0 = [-9.817, -9.596, -8.684, -8.434, -8.637, -8.710, -9.095, -9.219, -9.300, -9.512];
        // Frame 1
        let expected_f1 = [-6.475, -6.772, -7.163, -7.581, -7.956, -8.100, -7.900, -7.545, -7.380, -7.335];
        // Frame 2
        let expected_f2 = [-5.592, -6.123, -7.055, -7.128, -6.862, -7.186, -7.461, -6.631, -6.126, -6.012];

        for (i, &exp) in expected_f0.iter().enumerate() {
            let got = mel[i][0];
            assert!((got - exp).abs() < 1e-3, "Frame 0 mel[{}] = {:.3}, expected {:.3}", i, got, exp);
        }
        for (i, &exp) in expected_f1.iter().enumerate() {
            let got = mel[i][1];
            assert!((got - exp).abs() < 1e-3, "Frame 1 mel[{}] = {:.3}, expected {:.3}", i, got, exp);
        }
        for (i, &exp) in expected_f2.iter().enumerate() {
            let got = mel[i][2];
            assert!((got - exp).abs() < 1e-3, "Frame 2 mel[{}] = {:.3}, expected {:.3}", i, got, exp);
        }
    }
}