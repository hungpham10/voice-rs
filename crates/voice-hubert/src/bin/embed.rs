//! Example: Run HuBERT embedding on a WAV file.

use std::path::PathBuf;
use voice_hubert::wavio::{load_wav_mono, preprocess_applio, save_npy3};
use voice_hubert::HubertEmbedder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        // Default: generate a test sine wave
        PathBuf::from("test_sine.wav")
    });
    // Optional 2nd arg: write the `[1, T, 768]` features here as `.npy`.
    let out_path = args.get(2).map(PathBuf::from);

    println!("Loading HuBERT embedder...");
    let embedder = HubertEmbedder::new()?;
    println!("Device: {:?}", embedder.device());

    // Load audio
    let samples = if wav_path.exists() {
        println!("Loading audio from: {:?}", wav_path);
        load_wav_mono(&wav_path)?
    } else {
        println!("File not found, generating test sine wave (440 Hz, 2s)...");
        generate_sine_wave(440.0, 2.0, 16000)
    };

    println!("Input samples: {}", samples.len());
    println!("Sample rate: 16000 Hz");

    // Preprocess to match Applio's create_reference.py:
    // - peak normalize (for int PCM files)
    // - trim to multiple of 320
    // - reflect pad 40 samples on each side
    let samples = preprocess_applio(samples);

    println!("Preprocessed samples: {}", samples.len());

    // Run embedding
    println!("Running forward pass...");
    let embeddings = embedder.forward(&samples)?;

    let shape = embeddings.shape();
    let dims: [usize; 3] = shape.dims();
    println!("Output shape: {:?} (expected [1, T, 768])", dims);

    // Print basic statistics
    let data = embeddings.into_data();
    let values: Vec<f32> = data.as_slice::<f32>().unwrap().to_vec();
    let overall_mean = values.iter().sum::<f32>() / values.len() as f32;
    let overall_var = values.iter().map(|v| (v - overall_mean).powi(2)).sum::<f32>() / values.len() as f32;
    let overall_std = overall_var.sqrt();
    println!("Overall: mean={:.6}, std={:.6}", overall_mean, overall_std);

    // Persist features for downstream training (optional).
    if let Some(out) = &out_path {
        save_npy3(out, &values, dims)?;
        println!("Saved features: {} (shape {:?})", out.display(), dims);
    }

    // Compare with ground truth if available
    let gt_path = "sources/Applio/logs/reference/contentvec/feats.npy";
    if std::path::Path::new(gt_path).exists() {
        compare_with_ground_truth(&values, dims, gt_path)?;
    }

    Ok(())
}

fn compare_with_ground_truth(
    values: &[f32],
    dims: [usize; 3],
    gt_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check for NaN in output first
    if values.iter().any(|v| !v.is_finite()) {
        return Err("Output contains NaN or Inf — model weights likely not loaded correctly".into());
    }
    
    use std::fs::File;
    use std::io::{BufReader, Read, Seek, SeekFrom};
    
    // Parse .npy file
    let file_len = std::fs::metadata(gt_path)?.len();
    let mut file = File::open(gt_path)?;
    let mut reader = BufReader::new(&mut file);
    
    // Read magic string
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY" {
        return Err("Not a valid .npy file".into());
    }
    
    // Version
    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;
    
    // Header length
    let mut header_len_bytes = [0u8; 2];
    reader.read_exact(&mut header_len_bytes)?;
    let header_len = u16::from_le_bytes(header_len_bytes) as usize;

    // Skip header
    reader.seek(SeekFrom::Current(header_len as i64))?;

    // Guard: only compare when the ground truth has exactly our element count.
    // The reference `feats.npy` is a different recording; without this check the
    // reader would silently misalign and report a bogus "❌".
    let data_offset = (6 + 2 + 2 + header_len) as u64;
    let available = file_len.saturating_sub(data_offset) / 4;
    let expected_len = dims[0] * dims[1] * dims[2];
    if available as usize != expected_len {
        println!(
            "\n(ground truth has {available} frames vs our {expected_len} — different recording, skipping compare)"
        );
        return Ok(());
    }

    // Read data
    let mut buf = vec![0u8; expected_len * 4];
    reader.read_exact(&mut buf)?;
    
    let gt_values: Vec<f32> = buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    
    // Check for NaN in ground truth
    if gt_values.iter().any(|v| !v.is_finite()) {
        return Err("Ground truth contains NaN or Inf".into());
    }
    
    if gt_values.len() != values.len() {
        println!("  ⚠️  Length mismatch: got {}, expected {}", values.len(), gt_values.len());
        return Ok(());
    }
    
    let mut max_diff = 0.0f32;
    let mut mean_diff = 0.0f32;
    for (a, b) in values.iter().zip(gt_values.iter()) {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        mean_diff += diff;
    }
    mean_diff /= values.len() as f32;
    
    println!("\n=== Ground Truth Comparison ===");
    println!("  Max absolute diff: {:.6}", max_diff);
    println!("  Mean absolute diff: {:.6}", mean_diff);
    
    if max_diff < 1e-3 {
        println!("  ✅ Match within 1e-3");
    } else if max_diff < 1e-2 {
        println!("  ⚠️  Match within 1e-2");
    } else {
        println!("  ❌ Significant difference");
    }
    
    Ok(())
}

fn generate_sine_wave(freq_hz: f32, duration_s: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_s * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin())
        .collect()
}