//! Export f0 RMVPE cho một wav: `f0_export <in.wav> <out.f0.npy>` — @100 Hz.
//!
//! Dùng cho `voice-train diag --f0 <out.f0.npy>` (PR-B1: tách voiced/unvoiced).
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
        _ => panic!(
            "Unsupported WAV format: {} bits {:?}",
            spec.bits_per_sample, spec.sample_format
        ),
    };
    (samples, spec.sample_rate)
}

fn write_npy_f32(path: &Path, t: usize, flat: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut hdr = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': (1, {t}), }}"
    );
    while hdr.len() < 62 {
        hdr.push(' ');
    }
    hdr.push('\n');
    let mut f = std::fs::File::create(path)?;
    f.write_all(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0])?;
    f.write_all(&(hdr.len() as u16).to_le_bytes())?;
    f.write_all(hdr.as_bytes())?;
    f.write_all(&flat.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    Ok(())
}

fn main() -> Result<(), voice_mel::MelError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: f0_export <in.wav> <out.f0.npy>");
        std::process::exit(2);
    }
    let (samples, sr) = load_wav_mono(Path::new(&args[1]));
    if sr != 16000 {
        eprintln!("⚠️  wav {sr} Hz — RMVPE cần 16 kHz, kết quả lệch mel hop");
    }
    let mel_cfg = voice_mel::MelConfig::rmvpe();
    let mel = mel_cfg.spectrogram(&samples)?;
    let rmvpe = voice_f0::Rmvpe::new().expect("Failed to load RMVPE");
    let f0 = rmvpe.forward_f0(&mel).expect("Forward failed");
    println!("f0: {} frame @100 Hz → {}", f0.len(), args[2]);
    write_npy_f32(Path::new(&args[2]), f0.len(), &f0).expect("ghi f0.npy");
    Ok(())
}
