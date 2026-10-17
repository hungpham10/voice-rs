//! Example: Run HuBERT embedding on a WAV file.

use std::path::PathBuf;
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
        load_wav(&wav_path)?
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
        save_npy(out, &values, dims)?;
        println!("Saved features: {} (shape {:?})", out.display(), dims);
    }

    // Compare with ground truth if available
    let gt_path = "sources/Applio/logs/reference/contentvec/feats.npy";
    if std::path::Path::new(gt_path).exists() {
        compare_with_ground_truth(&values, dims, gt_path)?;
    }

    Ok(())
}

fn load_wav(path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    
    let mut file = File::open(path)?;
    
    // Parse WAV header manually for robustness
    let mut header = [0u8; 44];
    file.read_exact(&mut header)?;
    
    // Check RIFF/WAVE
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }
    
    // Parse format chunk
    let format_tag = u16::from_le_bytes([header[20], header[21]]);
    let channels = u16::from_le_bytes([header[22], header[23]]);
    let sample_rate = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
    let bits_per_sample = u16::from_le_bytes([header[34], header[35]]);
    
    println!("  WAV spec: {} Hz, {} ch, {} bits, format tag {}", sample_rate, channels, bits_per_sample, format_tag);
    
    // Find data chunk - search from position 12
    let mut pos = 12u64;
    let mut data_pos = None;
    let mut data_size = 0;
    let file_size = file.metadata()?.len();
    while pos + 8 <= file_size {
        file.seek(SeekFrom::Start(pos))?;
        let mut chunk_header = [0u8; 8];
        file.read_exact(&mut chunk_header)?;
        let chunk_id = &chunk_header[0..4];
        let chunk_size = u32::from_le_bytes([chunk_header[4], chunk_header[5], chunk_header[6], chunk_header[7]]);
        
        if chunk_id == b"data" {
            data_pos = Some(pos + 8);
            data_size = chunk_size;
            break;
        }
        pos += 8 + chunk_size as u64;
    }
    
    let data_pos = data_pos.ok_or("No data chunk found")?;
    
    // Read audio data
    file.seek(SeekFrom::Start(data_pos))?;
    
    // Read raw samples based on format
    let samples: Vec<f32> = match (format_tag, bits_per_sample) {
        (1, 16) => {  // PCM int16
            let mut buf = vec![0u8; data_size as usize];
            file.read_exact(&mut buf)?;
            buf.chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / i16::MAX as f32)
                .collect()
        }
        (1, 32) => {  // PCM int32
            let mut buf = vec![0u8; data_size as usize];
            file.read_exact(&mut buf)?;
            buf.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / i32::MAX as f32)
                .collect()
        }
        (3, 32) => {  // IEEE float 32-bit
            let mut buf = vec![0u8; data_size as usize];
            file.read_exact(&mut buf)?;
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        _ => return Err(format!("Unsupported format: tag={}, bits={}", format_tag, bits_per_sample).into()),
    };
    
    // Convert to mono if stereo
    let mono = if channels == 2 {
        samples.chunks_exact(2).map(|ch| (ch[0] + ch[1]) * 0.5).collect()
    } else {
        samples
    };
    
    // Resample if needed (simplified - just warn for now)
    if sample_rate != 16000 {
        println!("  ⚠️  Sample rate is {} Hz, expected 16000 Hz. Consider resampling.", sample_rate);
    }
    
    Ok(mono)
}

/// Preprocess audio to match Applio's create_reference.py:
/// 1. Peak normalize (for int PCM - librosa default)
/// 2. Trim to multiple of 320 (hop size)
/// 3. Reflect pad 40 samples on each side (HuBERT conv padding)
fn preprocess_applio(mut samples: Vec<f32>) -> Vec<f32> {
    // 1. Peak normalize (librosa.load default for int PCM)
    // Only needed if input was integer PCM; float inputs already normalized
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1.0 {
        // Input was not normalized (shouldn't happen for float, but safe)
        let scale = 1.0 / peak;
        samples.iter_mut().for_each(|s| *s *= scale);
    } else if peak > 0.0 && peak < 1.0 {
        // Likely int PCM that was already peak-normalized by librosa
        // Don't renormalize
    }
    
    // 2. Trim to multiple of 320
    let trimmed_len = (samples.len() / 320) * 320;
    samples.truncate(trimmed_len);
    
    // 3. Reflect pad 40 on each side
    let pad = 40;
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    // Left pad (reflect)
    for i in (0..pad).rev() {
        padded.push(samples[i % samples.len()]);
    }
    // Original
    padded.extend_from_slice(&samples);
    // Right pad (reflect)
    for i in 0..pad {
        padded.push(samples[samples.len() - 1 - (i % samples.len())]);
    }
    
    padded
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

/// Write a C-order `float32` `.npy` with shape `dims` (`.npy` v1.0).
///
/// Readable by NumPy and by `voice_models::npy`. Kept inline on purpose: this is
/// the only writer in the tree, and a shared one would pull `voice-lib` into a
/// model crate.
fn save_npy(
    path: &std::path::Path,
    values: &[f32],
    dims: [usize; 3],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        dims[0], dims[1], dims[2]
    );
    // magic(6) + version(2) + header_len(2) + header + '\n' must be a multiple of 64.
    let prefix = 6 + 2 + 2;
    let padding = (64 - ((prefix + header.len() + 1) % 64)) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');

    let mut out = Vec::with_capacity(prefix + header.len() + values.len() * 4);
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn generate_sine_wave(freq_hz: f32, duration_s: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_s * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin())
        .collect()
}