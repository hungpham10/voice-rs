//! Example: Compute mel spectrogram for a WAV file.

use std::path::PathBuf;
use voice_mel::MelConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("sources/Applio/logs/reference/reference.wav")
    });

    println!("Loading audio from: {:?}", wav_path);
    let samples = load_wav(&wav_path)?;

    println!("Input samples: {}, Sample rate: 16000 Hz", samples.len());

    // Compute mel spectrogram using RMVPE config
    let config = MelConfig::rmvpe();
    println!("\nConfig: {:?}", config);

    println!("\nComputing mel spectrogram...");
    let mel = config.spectrogram(&samples)?;

    let n_mels = mel.len();
    let n_frames = mel[0].len();

    println!("Output shape: [{} mel, {} frames]", n_mels, n_frames);
    println!("Frame rate: {:.1} Hz", 16000.0 / config.hop_length as f32);

    // Print statistics
    let all_values: Vec<f32> = mel.iter().flatten().copied().collect();
    let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
    let variance = all_values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / all_values.len() as f32;
    let std = variance.sqrt();
    let min = all_values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = all_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("\nStatistics:");
    println!("  Mean: {:.6}", mean);
    println!("  Std:  {:.6}", std);
    println!("  Min:  {:.6}", min);
    println!("  Max:  {:.6}", max);

    // Print first few frames
    println!("\nFirst 3 frames (first 10 mels each):");
    for j in 0..3.min(n_frames) {
        let frame: Vec<f32> = (0..10.min(n_mels)).map(|m| mel[m][j]).collect();
        println!("  Frame {}: {:?}", j, frame.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>());
    }

    Ok(())
}

fn load_wav(path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    let mut file = File::open(path)?;

    let mut header = [0u8; 44];
    file.read_exact(&mut header)?;

    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }

    let format_tag = u16::from_le_bytes([header[20], header[21]]);
    let channels = u16::from_le_bytes([header[22], header[23]]);
    let sample_rate = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
    let bits_per_sample = u16::from_le_bytes([header[34], header[35]]);

    println!("  WAV spec: {} Hz, {} ch, {} bits, format tag {}", sample_rate, channels, bits_per_sample, format_tag);

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
    file.seek(SeekFrom::Start(data_pos))?;

    let samples: Vec<f32> = match (format_tag, bits_per_sample) {
        (1, 16) => {
            let mut buf = vec![0u8; data_size as usize];
            file.read_exact(&mut buf)?;
            buf.chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / i16::MAX as f32)
                .collect()
        }
        (1, 32) => {
            let mut buf = vec![0u8; data_size as usize];
            file.read_exact(&mut buf)?;
            buf.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / i32::MAX as f32)
                .collect()
        }
        (3, 32) => {
            let mut buf = vec![0u8; data_size as usize];
            file.read_exact(&mut buf)?;
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        _ => return Err(format!("Unsupported format: tag={}, bits={}", format_tag, bits_per_sample).into()),
    };

    let mono = if channels == 2 {
        samples.chunks_exact(2).map(|ch| (ch[0] + ch[1]) * 0.5).collect()
    } else {
        samples
    };

    Ok(mono)
}