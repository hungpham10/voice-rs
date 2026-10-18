//! Nạp dữ liệu huấn luyện: wav nguồn 16 kHz và features HuBERT đã xuất.

use std::path::Path;

use burn::tensor::{Device, Tensor, TensorData};

/// Đọc một wav **mono** thành `Vec<f32>` (đã chuẩn hoá về `[-1, 1]`).
///
/// Chỉ hỗ trợ PCM 16/32-bit và IEEE float 32-bit — đủ cho dữ liệu hiện có. Trả về
/// kèm cảnh báo nếu sample rate khác 16 kHz (người gọi tự quyết định).
pub fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| Ok(s? as f32 / i16::MAX as f32))
            .collect::<Result<_, hound::Error>>()?,
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| Ok(s? as f32 / i32::MAX as f32))
            .collect::<Result<_, hound::Error>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, hound::Error>>()?,
        (fmt, bits) => {
            return Err(format!("wav không hỗ trợ: {fmt:?} {bits}-bit").into());
        }
    };

    let mono = if spec.channels == 2 {
        samples.chunks_exact(2).map(|c| (c[0] + c[1]) * 0.5).collect()
    } else {
        samples
    };

    Ok((mono, spec.sample_rate))
}

/// Nạp features HuBERT đã xuất thành tensor **channel-major** `[1, 768, T]`.
///
/// File `.npy` được `embed <wav> <out.npy>` ghi ở dạng time-major `[1, T, 768]`;
/// `Adapter` cần channel-major `[1, C, T]` nên ở đây hoán vị hai chiều cuối.
pub fn features_channel_major(
    path: &Path,
    device: &Device,
) -> Result<Tensor<3>, Box<dyn std::error::Error>> {
    let (shape, flat) = voice_vits::npy::read_f32(path)?;
    if shape.len() != 3 || shape[0] != 1 {
        return Err(format!("features phải là [1, T, C], được {shape:?}").into());
    }
    let (t, c) = (shape[1], shape[2]);
    let time_major = Tensor::<3>::from_data(TensorData::new(flat, [1, t, c]), device);
    Ok(time_major.swap_dims(1, 2))
}

/// Ghi wav float32 mono 22.05 kHz (nghe thử kết quả huấn luyện).
pub fn write_wav_mono(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}
