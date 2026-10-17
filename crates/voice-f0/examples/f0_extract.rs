//! Example: Extract F0 from reference.wav and compare with ground truth.

use std::path::Path;
use voice_f0::{Rmvpe, quantize};
use voice_mel::{MelConfig, MelError};

fn load_wav_mono(path: &Path) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV");
    let spec = reader.spec();
    println!("  WAV spec: channels={}, sample_rate={}, bits_per_sample={}, sample_format={:?}", 
             spec.channels, spec.sample_rate, spec.bits_per_sample, spec.sample_format);
    
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

fn load_npy_f32(path: &Path) -> Vec<f32> {
    let data = std::fs::read(path).expect("Failed to read npy");
    let npy = npy::NpyData::<f64>::from_bytes(&data).expect("Failed to parse npy as f64");
    println!("  npy len: {}", npy.len());
    npy.to_vec().into_iter().map(|x| x as f32).collect()
}

fn main() -> Result<(), MelError> {
    let reference_wav = Path::new("sources/Applio/logs/reference/reference.wav");
    let pitch_fine_npy = Path::new("sources/Applio/logs/reference/pitch_fine.npy");

    println!("Loading reference.wav...");
    let (samples, sr) = load_wav_mono(reference_wav);
    println!("  sample rate: {}, samples: {}", sr, samples.len());

    println!("Loading pitch_fine.npy (ground truth)...");
    let gt_f0 = load_npy_f32(pitch_fine_npy);
    println!("  ground truth frames: {}", gt_f0.len());

    println!("Computing mel spectrogram (RMVPE config)...");
    let mel_config = MelConfig::rmvpe();
    let mel = mel_config.spectrogram(&samples)?;
    println!("  mel shape: [{}][{}]", mel.len(), mel[0].len());

    println!("Loading RMVPE model...");
    let rmvpe = Rmvpe::new().expect("Failed to load RMVPE");

    println!("Running forward + decode...");
    let f0 = rmvpe.forward_f0(&mel).expect("Forward failed");
    println!("  predicted frames: {}", f0.len());

    // Compare
    if f0.len() != gt_f0.len() {
        println!("WARNING: frame count mismatch! pred={}, gt={}", f0.len(), gt_f0.len());
    }

    let n = f0.len().min(gt_f0.len());
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut n_voiced_pred = 0;
    let mut n_voiced_gt = 0;
    let mut n_voiced_both = 0;

    for i in 0..n {
        let diff = (f0[i] - gt_f0[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
        sum_diff += diff;

        if gt_f0[i] > 0.0 {
            let rel = diff / gt_f0[i];
            if rel > max_rel_diff {
                max_rel_diff = rel;
            }
        }

        let pred_voiced = f0[i] > 0.0;
        let gt_voiced = gt_f0[i] > 0.0;
        if pred_voiced { n_voiced_pred += 1; }
        if gt_voiced { n_voiced_gt += 1; }
        if pred_voiced && gt_voiced { n_voiced_both += 1; }
    }

    let mean_diff = sum_diff / n as f32;

    println!("\n=== Comparison Results ===");
    println!("Frames compared: {}", n);
    println!("Max abs diff:    {:.6} at frame {}", max_diff, max_diff_idx);
    println!("Mean abs diff:   {:.6}", mean_diff);
    println!("Max rel diff:    {:.6}", max_rel_diff);
    println!("Voiced frames:   pred={}, gt={}, both={}", n_voiced_pred, n_voiced_gt, n_voiced_both);

    // Print first 20 frames for visual inspection
    println!("\nFirst 20 frames:");
    for i in 0..20.min(n) {
        println!("  [{:3}] pred={:8.2}  gt={:8.2}  diff={:.4}", i, f0[i], gt_f0[i], (f0[i] - gt_f0[i]).abs());
    }

    // Print frames around max diff
    if max_diff_idx > 20 {
        println!("\nFrames around max diff (idx={}):", max_diff_idx);
        let start = max_diff_idx.saturating_sub(5);
        let end = (max_diff_idx + 5).min(n);
        for i in start..end {
            println!("  [{:3}] pred={:8.2}  gt={:8.2}  diff={:.4}", i, f0[i], gt_f0[i], (f0[i] - gt_f0[i]).abs());
        }
    }

    // Also test quantize
    let coarse = quantize(&f0);
    println!("\nCoarse (first 20): {:?}", &coarse[..20.min(n)]);

    // Load pitch_coarse.npy (int32)
    let pitch_coarse_npy = Path::new("sources/Applio/logs/reference/pitch_coarse.npy");
    if pitch_coarse_npy.exists() {
        let data = std::fs::read(pitch_coarse_npy).expect("Failed to read pitch_coarse.npy");
        let npy = npy::NpyData::<i32>::from_bytes(&data).expect("Failed to parse pitch_coarse.npy as i32");
        println!("  pitch_coarse len: {}", npy.len());
        let gt_coarse: Vec<f32> = npy.to_vec().into_iter().map(|x| x as f32).collect();
        println!("Ground truth coarse (first 20): {:?}", &gt_coarse[..20.min(gt_coarse.len())]);
    }

    Ok(())
}